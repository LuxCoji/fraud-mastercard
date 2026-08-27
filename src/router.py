"""Route a transaction to its rail, score it, and decide what to do.

Two models exist and nothing joined them. This does.

Routing is **deterministic, from the payload** - never learned. A transaction
arrives on a known rail and the payload says which; asking a model to infer it
would add latency and a failure mode for no benefit.

The rails are deliberately not symmetric, because their economics are not:

| | card | wire |
| --- | --- | --- |
| base rate | 3.44% | 0.17% |
| precision at the operating point | 78% | ~10% |
| decision | allow / step up / block | allow / investigate |

**The card rail decides. The wire rail recommends.** At 10% precision, blocking
on the model's say-so would freeze roughly nine innocent parties per real one -
so the wire rail never blocks, it queues a case for a human. That difference is
encoded here rather than left to whoever calls this.

One more asymmetry worth stating: the card rail runs *inside* the authorisation
window (measured 5.53 ms p99 against a 100 ms budget), while the wire rail runs
asynchronously and can afford graph work that would never fit in-line.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class Rail(str, Enum):
    CARD = "card"
    WIRE = "wire"
    TRANSFER = "transfer"


class Action(str, Enum):
    ALLOW = "allow"
    STEP_UP = "step_up"        # challenge the customer - an OTP, not a decline
    BLOCK = "block"
    INVESTIGATE = "investigate"  # queue for a human; never an automatic freeze


@dataclass
class Decision:
    """What the system decided, and enough context to explain it."""

    rail: Rail
    action: Action
    score: float
    reason: str
    scored: bool = True
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"rail": self.rail.value, "action": self.action.value,
                "score": round(self.score, 6), "reason": self.reason,
                "scored": self.scored, "evidence": self.evidence}


# Thresholds are expressed as the share of traffic each band takes, then
# converted to score cut-offs against a scored sample. A business can reason
# about "we decline one transaction in two hundred"; it cannot reason about 0.9167.
CARD_BLOCK_RATE = 0.005      # top 0.5% - measured 95.3% precision there
CARD_ALERT_RATE = 0.02       # everything above this is touched at all
WIRE_ALERT_RATE = 0.01       # investigator capacity, not a confidence level

# PaySim's fraud is confined to two payment types, but that is a property of the
# *simulator* rather than of payments - see reports/step3. Nothing here treats a
# payment type as safe.
TRANSFER_RAIL_ENABLED = False


def route(txn: dict) -> Rail:
    """Which rail does this transaction belong to?

    Read from the payload. A card authorisation carries card fields; a wire
    carries bank and account fields. No inference, no model.
    """
    if "TransactionAmt" in txn and "card1" in txn:
        return Rail.CARD
    if "Amount Paid" in txn and "From Bank" in txn:
        return Rail.WIRE
    if "type" in txn and "nameOrig" in txn:
        return Rail.TRANSFER
    raise ValueError("cannot route: payload matches no known rail")


class FraudRouter:
    """One entry point for a transaction of any rail.

    Scorers are injected rather than loaded here so this stays testable and so a
    caller can run with only the rails it has models for.
    """

    def __init__(self, card_scorer=None, wire_scorer=None,
                 card_block_threshold: float | None = None,
                 card_alert_threshold: float | None = None,
                 wire_alert_threshold: float | None = None) -> None:
        self.card_scorer = card_scorer
        self.wire_scorer = wire_scorer
        self.card_block_threshold = card_block_threshold
        self.card_alert_threshold = card_alert_threshold
        self.wire_alert_threshold = wire_alert_threshold

    # -- per-rail policy ----------------------------------------------------
    def _decide_card(self, score: float) -> Decision:
        """Three bands. Blocking is reserved for near-certainty.

        Measured on the held-out test set: a single threshold at 2% declines
        0.456% of genuine customers; splitting the same budget so only the top
        0.5% is blocked declines 0.025% - **95% fewer** - and catches the same
        amount of fraud, because the difference is challenged rather than lost.
        """
        if self.card_block_threshold is not None and score >= self.card_block_threshold:
            return Decision(Rail.CARD, Action.BLOCK, score,
                            "score in the top band, where precision is ~95%")
        if self.card_alert_threshold is not None and score >= self.card_alert_threshold:
            return Decision(Rail.CARD, Action.STEP_UP, score,
                            "elevated risk - challenge rather than decline")
        return Decision(Rail.CARD, Action.ALLOW, score, "below the alert threshold")

    def _decide_wire(self, score: float) -> Decision:
        """Two bands, and neither of them blocks.

        Precision at the wire operating point is around 10%, so an automatic
        freeze would stop roughly nine innocent parties per real one. The output
        is a case for a human, which is how AML works in practice.
        """
        if self.wire_alert_threshold is not None and score >= self.wire_alert_threshold:
            return Decision(Rail.WIRE, Action.INVESTIGATE, score,
                            "queued for review - this rail never blocks automatically")
        return Decision(Rail.WIRE, Action.ALLOW, score, "below the alert threshold")

    # -- entry point --------------------------------------------------------
    def decide(self, txn: dict) -> Decision:
        rail = route(txn)

        if rail is Rail.CARD:
            if self.card_scorer is None:
                return Decision(rail, Action.ALLOW, 0.0,
                                "no card model loaded", scored=False)
            return self._decide_card(self.card_scorer.score(txn))

        if rail is Rail.WIRE:
            if self.wire_scorer is None:
                return Decision(rail, Action.ALLOW, 0.0,
                                "no wire model loaded", scored=False)
            return self._decide_wire(self.wire_scorer.score(txn))

        # The transfer rail has no trustworthy model: PaySim's fraud is
        # separable by a three-line rule that holds for 97.8% of fraud and
        # 0.00% of legitimate traffic, which is the simulator showing through.
        # Allowing unscored is the honest behaviour - a fabricated score would
        # be worse than none.
        return Decision(rail, Action.ALLOW, 0.0,
                        "transfer rail unvalidated - no public data supports a model",
                        scored=False)


def thresholds_from_scores(scores, block_rate: float = CARD_BLOCK_RATE,
                           alert_rate: float = CARD_ALERT_RATE) -> tuple[float, float]:
    """Convert traffic shares into score cut-offs using a scored sample.

    Thresholds are chosen by rate rather than by score because the rate is the
    business decision. The score that implements it is an artefact of the model
    and changes whenever the model is retrained.
    """
    import numpy as np
    ordered = np.sort(np.asarray(scores))[::-1]
    n = len(ordered)
    return (float(ordered[max(0, int(round(n * block_rate)) - 1)]),
            float(ordered[max(0, int(round(n * alert_rate)) - 1)]))
