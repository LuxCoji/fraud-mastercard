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

# The step-up threshold varies with the amount at risk; the block threshold does
# not. Elkan (2001) puts the cost-minimising boundary at
#
#     threshold = cost(false positive) / (cost(false positive) + cost(false negative))
#
# where the false-negative cost is the amount and the false-positive cost is
# roughly fixed. So a large transaction should need less confidence to challenge.
#
# The exponent damps that. Measured at an identical 2% alert budget:
#
#     exponent  fraud value stopped  customers flagged  of the 20 largest
#     0.0 (flat)        30.7%              511                 1
#     0.5               44.2%            1,078 (2.1x)          9
#     1.0 (full)        45.4%            1,552 (3.0x)         10
#
# 0.5 is the knee - most of the value gain for 2.1x the friction rather than 3x.
# Past it the gain flattens while the cost keeps climbing.
AMOUNT_EXPONENT = 0.5

# **Applied to step-up only.** A challenge is an OTP - cheap friction, and worth
# spending on a large transaction. A block is a lost sale and an angry customer,
# so it keeps a fixed high bar regardless of amount. Making blocking easier for
# large transactions would put the system's most expensive mistake exactly where
# the customer is most valuable.
AMOUNT_AWARE_STEP_UP = True

# Geography shifts the step-up threshold too, but for a different reason than
# amount: it changes the *prior* probability of fraud rather than the cost of
# being wrong. Measured on the held-out card period:
#
#     domestic        89.5% of traffic,  2.28% fraud,  2,414 frauds
#     cross-border     0.4% of traffic,  8.77% fraud,     37 frauds
#     country unknown 10.1% of traffic, 13.50% fraud,  1,613 frauds
#
# The finding worth noting is that **a missing country code, not a foreign one,
# is where the risk lives**. It is 5.9x the domestic rate and carries 40% of all
# fraud in a tenth of traffic. Missingness is the signal here, not a data
# quality nuisance.
#
# **Tested and switched off.** Adjusting the threshold by segment *lost* 59% of
# the value recall, and the reason is worth keeping:
#
#     segment       fraud rate    mean score    score ratio
#     domestic          2.283%        0.0165          1.00x
#     cross-border      8.768%        0.0918          5.58x
#     unknown          13.504%        0.1338          8.13x
#
# The model already scores unknown-country transactions 8.13x higher than
# domestic ones, against a 5.91x difference in actual fraud rate. It has priced
# the segment in - it slightly *over*-prices it - because addr2 is one of its
# input features. Lowering the threshold on top of that counts the same evidence
# twice.
#
# The distinction that matters, and the reason the amount adjustment does work:
#
#     amount  changes the COST of a mistake     - the model cannot know this
#     segment changes the PROBABILITY of fraud  - the model already knows this
#
# A threshold should only be adjusted for things outside the model's view.
# Anything the features already capture belongs in the score, not in the policy.
SEGMENT_AWARE_STEP_UP = False


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
                 wire_alert_threshold: float | None = None,
                 amount_cost: float | None = None,
                 segment_risk: dict | None = None,
                 home_country=None) -> None:
        self.card_scorer = card_scorer
        self.wire_scorer = wire_scorer
        self.card_block_threshold = card_block_threshold
        self.card_alert_threshold = card_alert_threshold
        self.wire_alert_threshold = wire_alert_threshold
        # Calibrated so the amount-aware step-up spends the same alert budget as
        # a flat threshold would. None disables it and restores flat behaviour.
        self.amount_cost = amount_cost
        # Risk multiple per geography segment, fitted on the training period.
        self.segment_risk = segment_risk or {}
        self.home_country = home_country

    def step_up_threshold(self, amount: float | None,
                          addr2=None) -> float:
        """The confidence this transaction needs to be challenged.

        Two adjustments, both downward and both bounded:

        **Amount** - Elkan's rule. A larger amount at risk justifies acting on
        less confidence, because being wrong costs the same either way.

        **Segment** - geography. A segment whose observed fraud rate is 5.9x the
        domestic baseline needs proportionally less confidence, because the
        prior is already against it.

        Bounded at both ends. Without a floor a large enough amount would
        challenge every transaction of that size regardless of score; without a
        ceiling a near-zero amount would never be challenged at all.
        """
        flat = self.card_alert_threshold
        threshold = flat if flat is not None else 1.0

        if AMOUNT_AWARE_STEP_UP and self.amount_cost is not None and amount is not None:
            threshold = self.amount_cost / (
                self.amount_cost + max(float(amount), 0.0) ** AMOUNT_EXPONENT)

        if SEGMENT_AWARE_STEP_UP and self.segment_risk:
            from policy.segments import segment_of
            multiple = self.segment_risk.get(
                segment_of(addr2, self.home_country), 1.0)
            threshold /= multiple

        return float(min(max(threshold, 0.01), 0.99))

    # -- per-rail policy ----------------------------------------------------
    def _decide_card(self, score: float, amount: float | None = None,
                     addr2=None) -> Decision:
        """Three bands. Blocking is reserved for near-certainty.

        Measured on the held-out test set: a single threshold at 2% declines
        0.456% of genuine customers; splitting the same budget so only the top
        0.5% is blocked declines 0.025% - **95% fewer** - and catches the same
        amount of fraud, because the difference is challenged rather than lost.

        The block threshold is fixed. The step-up threshold varies with amount.
        """
        if self.card_block_threshold is not None and score >= self.card_block_threshold:
            return Decision(Rail.CARD, Action.BLOCK, score,
                            "score in the top band, where precision is ~95%",
                            evidence={"threshold": self.card_block_threshold,
                                      "amount_aware": False})

        threshold = self.step_up_threshold(amount, addr2)
        if score >= threshold:
            aware = (AMOUNT_AWARE_STEP_UP and self.amount_cost is not None
                     and amount is not None)
            reason = ("elevated risk for this amount - challenge rather than decline"
                      if aware else "elevated risk - challenge rather than decline")
            return Decision(Rail.CARD, Action.STEP_UP, score, reason,
                            evidence={"threshold": round(threshold, 6),
                                      "amount": amount, "amount_aware": aware})
        return Decision(Rail.CARD, Action.ALLOW, score, "below the alert threshold",
                        evidence={"threshold": round(threshold, 6)})

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
            return self._decide_card(self.card_scorer.score(txn),
                                     txn.get("TransactionAmt"),
                                     txn.get("addr2"))

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
