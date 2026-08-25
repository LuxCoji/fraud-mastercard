"""Three-band decision policy: allow, step up, block.

A single threshold forces every flag to be a hard decline, which is the most
expensive mistake available - a lost sale plus an angry customer. But a
step-up challenge (an OTP, say) is cheap: mild friction for a genuine customer,
and a real obstacle for someone holding a stolen card.

So the policy uses two thresholds rather than one:

    score >= block_threshold      -> block      (reserved for near-certainty)
    score >= step_up_threshold    -> step up    (the ambiguous middle)
    otherwise                     -> allow

The point is to spend the *block* budget only where precision is very high, and
push everything else into the step-up band. That collapses the number of genuine
customers who are hard-declined without giving up recall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALLOW, STEP_UP, BLOCK = "allow", "step_up", "block"


@dataclass
class Policy:
    """Two score thresholds and the rates they were chosen to hit."""

    block_threshold: float
    step_up_threshold: float
    block_rate: float
    step_up_rate: float

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """Map scores to decisions."""
        out = np.full(len(scores), ALLOW, dtype=object)
        out[scores >= self.step_up_threshold] = STEP_UP
        out[scores >= self.block_threshold] = BLOCK
        return out


def fit_policy(scores: np.ndarray, block_rate: float,
               total_alert_rate: float) -> Policy:
    """Choose thresholds by the share of traffic each band should receive.

    `block_rate` is the fraction of traffic hard-declined; `total_alert_rate` is
    the fraction touched at all, so the step-up band gets the difference.

    Rates are used rather than raw score values because a rate is something a
    business can actually reason about - "we decline one transaction in two
    hundred" - while a threshold of 0.83 means nothing on its own.
    """
    if block_rate > total_alert_rate:
        raise ValueError("block_rate cannot exceed total_alert_rate")

    ordered = np.sort(scores)[::-1]
    n = len(ordered)
    block_cut = ordered[max(0, int(round(n * block_rate)) - 1)]
    step_up_cut = ordered[max(0, int(round(n * total_alert_rate)) - 1)]

    return Policy(
        block_threshold=float(block_cut),
        step_up_threshold=float(step_up_cut),
        block_rate=block_rate,
        step_up_rate=total_alert_rate - block_rate,
    )


def evaluate_policy(y_true: np.ndarray, scores: np.ndarray,
                    policy: Policy) -> pd.DataFrame:
    """What each band actually catches and costs.

    `genuine_pct` is read per band against all genuine transactions, so the
    block row answers the question that matters most: what fraction of honest
    customers get turned away outright?
    """
    decisions = policy.decide(scores)
    total_fraud = int(y_true.sum())
    total_genuine = int((y_true == 0).sum())

    rows = []
    for band in (BLOCK, STEP_UP, ALLOW):
        mask = decisions == band
        fraud = int(y_true[mask].sum())
        genuine = int(mask.sum() - fraud)
        rows.append({
            "band": band,
            "transactions": int(mask.sum()),
            "share_of_traffic": mask.mean(),
            "fraud": fraud,
            "fraud_recall": fraud / total_fraud if total_fraud else 0.0,
            "genuine": genuine,
            "genuine_pct": genuine / total_genuine if total_genuine else 0.0,
            "precision": fraud / mask.sum() if mask.sum() else 0.0,
        })
    return pd.DataFrame(rows)


def format_policy(table: pd.DataFrame, policy: Policy) -> str:
    """Readable summary of a fitted policy."""
    lines = [
        f"block  at score >= {policy.block_threshold:.4f}",
        f"step up at score >= {policy.step_up_threshold:.4f}",
        "",
        f"{'band':>9} {'traffic':>9} {'fraud caught':>13} {'precision':>10} "
        f"{'genuine affected':>17}",
        "-" * 62,
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r.band:>9} {r.share_of_traffic:>8.2%} {r.fraud_recall:>12.1%} "
            f"{r.precision:>10.1%} {r.genuine_pct:>16.2%}"
        )
    return "\n".join(lines)
