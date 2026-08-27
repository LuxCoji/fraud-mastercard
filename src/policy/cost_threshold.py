"""Instance-dependent thresholds: cheap transactions face a higher bar.

The shipped card model catches 45.6% of frauds by count but only 30.7% by value,
and misses 17 of the 20 largest frauds. Investigating that showed the features
are not the problem - among large transactions alone, the best single feature
still separates fraud at AUC 0.731 against 0.747 overall. Large frauds are
visible; they simply fall below a threshold set for the average transaction.

Elkan's 2001 result gives the correction directly. Under a cost matrix the
optimal decision boundary is

    threshold* = cost(false positive) / (cost(false positive) + cost(false negative))

Here `cost(false negative)` is the amount at risk, which varies per transaction,
while `cost(false positive)` is roughly fixed - the friction of a challenge or
the lost margin on a decline. So the threshold is not a constant:

    threshold(amount) = c_fp / (c_fp + amount)

A 10,000-rupee transaction needs far less confidence to be worth stopping than a
50-rupee one, because the downside of being wrong about it is the same while the
downside of being right is two hundred times larger.

This needs no retraining. It is a decision rule applied to scores the model
already produces, which is what makes it worth trying before any modelling work.

The literature on instance-dependent cost-sensitive fraud detection reports the
same effect this is meant to produce: the average amount of *missed* fraud falls
sharply once thresholds vary with amount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def instance_thresholds(amounts: np.ndarray, false_positive_cost: float,
                        floor: float = 0.01, ceiling: float = 0.99) -> np.ndarray:
    """Per-transaction thresholds from Elkan's rule.

    `false_positive_cost` is expressed in the same units as `amounts` - what one
    wrongly flagged transaction costs. It is the only parameter, and it sets
    where the whole curve sits: a large value makes the system cautious
    everywhere, a small one makes it aggressive.

    Bounded at both ends. Without a floor, a large enough amount drives the
    threshold to zero and flags every transaction of that size regardless of
    score; without a ceiling, a near-zero amount never gets flagged at all.
    """
    amounts = np.asarray(amounts, dtype="float64")
    thresholds = false_positive_cost / (false_positive_cost + np.maximum(amounts, 0))
    return np.clip(thresholds, floor, ceiling)


def calibrate_fp_cost(scores: np.ndarray, amounts: np.ndarray,
                      target_alert_rate: float,
                      lo: float = 0.1, hi: float = 1e7,
                      tolerance: float = 1e-4) -> float:
    """Find the false-positive cost that produces a given alert rate.

    The alert rate is the operational constraint - review capacity is fixed -
    so it is more useful to ask "what cost assumption spends exactly my budget"
    than to guess a cost and accept whatever volume falls out.

    Binary search: the alert rate falls monotonically as the assumed cost rises,
    because a higher false-positive cost raises every threshold.
    """
    for _ in range(200):
        mid = (lo + hi) / 2
        rate = float((scores >= instance_thresholds(amounts, mid)).mean())
        if abs(rate - target_alert_rate) < tolerance:
            return mid
        if rate > target_alert_rate:
            lo = mid          # flagging too much; the cost must be higher
        else:
            hi = mid
    return (lo + hi) / 2


def compare(y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
            alert_rate: float = 0.02, fp_cost: float | None = None) -> pd.DataFrame:
    """Fixed threshold against instance-dependent, at the same alert rate.

    Holding the alert rate equal is what makes the comparison fair: both
    policies touch the same share of traffic, so any difference is in *which*
    transactions they pick rather than how many.
    """
    fixed_threshold = float(np.quantile(scores, 1 - alert_rate))
    fixed_flagged = scores >= fixed_threshold

    if fp_cost is None:
        fp_cost = calibrate_fp_cost(scores, amounts, alert_rate)
    variable_flagged = scores >= instance_thresholds(amounts, fp_cost)

    fraud = y_true == 1
    total_fraud_value = float(amounts[fraud].sum())
    largest = np.argsort(-amounts[fraud])[:20]

    rows = []
    for name, flagged in (("fixed threshold", fixed_flagged),
                          ("instance-dependent", variable_flagged)):
        caught = flagged & fraud
        rows.append({
            "policy": name,
            "alert_rate": float(flagged.mean()),
            "count_recall": float(caught.sum() / fraud.sum()),
            "value_recall": float(amounts[caught].sum() / total_fraud_value),
            "value_stopped": float(amounts[caught].sum()),
            "genuine_flagged": int((flagged & ~fraud).sum()),
            "genuine_value": float(amounts[flagged & ~fraud].sum()),
            "median_missed_fraud": float(np.median(amounts[fraud & ~flagged]))
            if (fraud & ~flagged).any() else 0.0,
            "top20_caught": int(flagged[fraud][largest].sum()),
        })
    out = pd.DataFrame(rows)
    out.attrs["fp_cost"] = fp_cost
    return out
