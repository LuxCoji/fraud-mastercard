"""Scoring metrics for fraud models.

AUC is reported because it is comparable to published numbers, but it is not
the metric this project optimises. What matters operationally is: at a review
capacity you can actually staff, how much fraud do you catch, and how many
genuine customers do you disturb?

`operating_points` answers that directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Fractions of traffic a review/step-up process might realistically absorb.
ALERT_RATES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]


def headline(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Ranking quality plus calibration quality.

    Brier is included next to a baseline because ranking metrics are blind to
    calibration: a model can rank perfectly while its probabilities are wildly
    wrong, and every policy threshold is derived from a probability.
    """
    base_rate = float(y_true.mean())
    return {
        "auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
        "brier_baseline": base_rate * (1 - base_rate),
        "base_rate": base_rate,
    }


def operating_points(y_true: np.ndarray, scores: np.ndarray,
                     alert_rates: list[float] = ALERT_RATES) -> pd.DataFrame:
    """For each alert rate, what does it buy and what does it cost?

    `alert_rate` is the fraction of all transactions flagged. `recall` is the
    share of fraud that lands inside those flags. `false_alarms_per_catch` is
    the number of genuine customers disturbed for each fraud caught - the
    number that decides whether an operating point is tolerable.
    """
    n = len(y_true)
    total_fraud = int(y_true.sum())
    order = np.argsort(-scores, kind="stable")
    labels_by_rank = y_true[order]

    rows = []
    for rate in alert_rates:
        k = max(1, int(round(n * rate)))
        flagged = labels_by_rank[:k]
        caught = int(flagged.sum())
        false_alarms = k - caught
        rows.append({
            "alert_rate": rate,
            "flagged": k,
            "fraud_caught": caught,
            "recall": caught / total_fraud if total_fraud else 0.0,
            "precision": caught / k,
            "false_alarms": false_alarms,
            "false_alarms_per_catch": false_alarms / caught if caught else np.inf,
            "genuine_disturbed_pct": false_alarms / (n - total_fraud),
        })
    return pd.DataFrame(rows)


def format_operating_points(table: pd.DataFrame) -> str:
    """Human-readable version of the operating-point table."""
    lines = [
        f"{'flag rate':>10} {'fraud caught':>13} {'precision':>10} "
        f"{'genuine hit':>12} {'false/catch':>12}",
        "-" * 62,
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r.alert_rate:>9.1%} {r.recall:>12.1%} {r.precision:>10.1%} "
            f"{r.genuine_disturbed_pct:>11.2%} {r.false_alarms_per_catch:>12.1f}"
        )
    return "\n".join(lines)


# --- value-weighted views ----------------------------------------------------
# Counting frauds treats a 500-rupee loss and a 500,000-rupee loss as one event
# each. The business does not. These functions weight by amount so the headline
# becomes "what share of fraud *losses* did we stop", which is the number a bank
# actually reports.
#
# The symmetry matters and is easy to lose: a wrongly blocked large payment is
# also a larger mistake than a wrongly blocked small one. Both sides are weighted
# here, so this is not an argument for blocking everything expensive.

def value_operating_points(y_true: np.ndarray, scores: np.ndarray,
                           amounts: np.ndarray,
                           alert_rates: list[float] = ALERT_RATES) -> pd.DataFrame:
    """Operating points measured in money rather than in transaction counts.

    `value_recall` is the share of total fraud *value* sitting inside the flagged
    set - the money the system had a chance to stop. `genuine_value_disturbed`
    is the counterpart cost: the value of legitimate payments caught up in it.
    """
    n = len(y_true)
    order = np.argsort(-scores, kind="stable")
    labels, values = y_true[order], amounts[order]

    total_fraud_value = float(values[labels == 1].sum())
    total_genuine_value = float(values[labels == 0].sum())

    rows = []
    for rate in alert_rates:
        k = max(1, int(round(n * rate)))
        top_labels, top_values = labels[:k], values[:k]

        caught_value = float(top_values[top_labels == 1].sum())
        disturbed_value = float(top_values[top_labels == 0].sum())
        rows.append({
            "alert_rate": rate,
            "count_recall": float((top_labels == 1).sum() / max((labels == 1).sum(), 1)),
            "value_recall": caught_value / total_fraud_value if total_fraud_value else 0.0,
            "fraud_value_stopped": caught_value,
            "genuine_value_disturbed": disturbed_value,
            "genuine_value_pct": disturbed_value / total_genuine_value if total_genuine_value else 0.0,
            # How many rupees of legitimate payment are touched per rupee of
            # fraud stopped. Below 1.0 the system is buying value cheaply.
            "value_cost_ratio": disturbed_value / caught_value if caught_value else np.inf,
        })
    return pd.DataFrame(rows)


def net_savings(y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                threshold: float, false_positive_cost: float = 0.0,
                fraud_recovery_rate: float = 1.0) -> dict:
    """Money saved, net of what the false positives cost.

    Two parameters the caller has to supply, because neither is a property of
    the model:

    `false_positive_cost` - what one wrongly flagged transaction costs. For a
    step-up challenge that is a small friction cost; for a hard block it is the
    lost sale plus the support call plus some churn risk. Left at zero by
    default so nothing is silently assumed.

    `fraud_recovery_rate` - what share of a *stopped* fraud is genuinely saved.
    Not always 1.0: some fraud would have been recovered anyway through
    chargebacks, in which case blocking it saves the dispute cost rather than
    the full amount.
    """
    flagged = scores >= threshold
    fraud, genuine = y_true == 1, y_true == 0

    stopped = float(amounts[flagged & fraud].sum()) * fraud_recovery_rate
    missed = float(amounts[~flagged & fraud].sum())
    false_positives = int((flagged & genuine).sum())
    fp_cost = false_positives * false_positive_cost

    return {
        "fraud_value_stopped": stopped,
        "fraud_value_missed": missed,
        "false_positives": false_positives,
        "false_positive_cost": fp_cost,
        "net_saving": stopped - fp_cost,
        "value_recall": stopped / (stopped + missed) if (stopped + missed) else 0.0,
    }


def format_value_points(table: pd.DataFrame) -> str:
    """Count recall against value recall, side by side.

    Showing both is the point: a gap between them means the model's ranking is
    not aligned with where the money is.
    """
    lines = [
        f"{'flag rate':>10} {'by count':>10} {'by value':>10} "
        f"{'value stopped':>16} {'genuine touched':>17}",
        "-" * 68,
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r.alert_rate:>9.1%} {r.count_recall:>10.1%} {r.value_recall:>10.1%} "
            f"{r.fraud_value_stopped:>16,.0f} {r.genuine_value_disturbed:>17,.0f}")
    return "\n".join(lines)
