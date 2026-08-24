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
