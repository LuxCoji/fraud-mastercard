"""Detect when the model has stopped working, before the labels say so.

A fraud model learns the patterns of the period it was trained on. Criminals
change tactics, and the model degrades quietly - nothing throws an error, the
API keeps returning scores, and the only symptom is that fewer frauds get caught.

**The hard part is timing.** Ground truth arrives weeks to months late:
chargebacks, disputes, investigations. Waiting for labels to tell you the model
broke means finding out a quarter after it happened.

So this watches two things:

  **Immediately available** - the score distribution, the alert rate, the input
  features. All computable the moment a transaction is scored.
  **Estimated performance** - a label-free estimate of what precision and recall
  *would* be, derived from the model's own calibrated confidence. This is the
  approach NannyML calls Confidence-Based Performance Estimation, and it works
  precisely because the card rail's scores are Platt-calibrated: a calibrated
  0.9 means the model is right about nine times in ten, so the expected
  confusion matrix can be summed from the scores alone.

CBPE is only valid while calibration holds. If the *relationship* between
features and fraud changes - concept drift proper - the estimate becomes
confidently wrong. That is a real limitation and it is why `label_backfill`
exists: when labels do arrive, the estimate is checked against them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Population Stability Index thresholds, the long-standing convention in credit
# scoring. Kept because they are widely understood, not because they are exact.
PSI_MINOR = 0.10        # worth logging
PSI_MAJOR = 0.25        # worth acting on

# How far the alert rate may move before it means something. The card rail's
# operating point is 2%; drifting to 3% is a 50% relative change and a real
# capacity problem.
ALERT_RATE_TOLERANCE = 0.25

# Features that are *supposed* to move as the calendar advances. A month index
# necessarily differs between one period and the next, and flagging that as
# drift means a CRITICAL report every single day - which trains everyone to
# ignore the monitor. Measured on two halves of the same held-out period with
# no drift at all: DT_M scored a PSI of 5.14, twenty times the "major" line.
TIME_INDEX_FEATURES = {"DT_M", "day", "day_of_week", "hour", "TransactionDT", "step"}


class Severity(str):
    OK = "ok"
    WATCH = "watch"        # log it
    WARN = "warn"          # tell someone
    CRITICAL = "critical"  # retrain


@dataclass
class DriftFinding:
    metric: str
    severity: str
    baseline: float
    current: float
    detail: str = ""

    @property
    def change(self) -> float:
        return (self.current - self.baseline) / self.baseline if self.baseline else np.nan

    def describe(self) -> str:
        return (f"[{self.severity:>8}] {self.metric:<28} "
                f"{self.baseline:>10.4f} -> {self.current:>10.4f} "
                f"({self.change:+.1%})  {self.detail}")


def build_bins(values: np.ndarray, bins: int = 10) -> tuple[list[float], list[float]]:
    """Bin edges from a reference sample, plus the proportion in each.

    Returns what a baseline needs to store. Edges come from the reference
    quantiles so the question is always "where do today's values fall in
    yesterday's buckets".
    """
    values = values[np.isfinite(values)]
    if len(values) < bins:
        return [], []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return [], []
    edges[0], edges[-1] = -np.inf, np.inf
    proportions = np.histogram(values, bins=edges)[0] / len(values)
    return edges.tolist(), proportions.tolist()


def psi_from_bins(edges: list[float], reference: list[float],
                  current: np.ndarray) -> float:
    """Population stability index against stored bins.

    **Bins are stored, not reconstructed.** An earlier version kept eleven
    quantiles and rebuilt the reference distribution by interpolating between
    them. That is adequate for a smooth feature and badly wrong for a
    concentrated one: the C columns are counts, mostly small integers, and
    interpolation turns a spike at zero into a continuous ramp. Two halves of
    the *same* period with no drift scored PSI 3.60 on P_emaildomain_count and
    2.37 on C9 - eight of twenty features over the "major" line, purely from
    the reconstruction.
    """
    current = current[np.isfinite(current)]
    if not edges or len(current) < 10:
        return 0.0
    c = np.histogram(current, bins=np.array(edges))[0] / len(current)
    b = np.array(reference)
    # A zero bucket makes the log infinite; a small floor keeps it finite.
    floor = 1e-6
    b, c = np.maximum(b, floor), np.maximum(c, floor)
    return float(np.sum((c - b) * np.log(c / b)))


def population_stability_index(baseline: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between two samples. Convenience wrapper over the two functions above."""
    edges, proportions = build_bins(baseline, bins)
    return psi_from_bins(edges, proportions, current)


def estimate_performance(scores: np.ndarray, threshold: float) -> dict:
    """Expected precision and recall from calibrated scores, with no labels.

    For a calibrated model, a score *is* the probability of fraud. So over the
    flagged set, the expected number of true positives is the sum of their
    scores, and the expected false positives is the sum of (1 - score). No
    ground truth required.

    This is only as good as the calibration. It will not notice concept drift -
    a model can be confidently wrong - which is why it is paired with the
    distribution checks above and confirmed by `label_backfill` later.
    """
    flagged = scores >= threshold
    if not flagged.any():
        return {"alert_rate": 0.0, "expected_precision": np.nan,
                "expected_recall": np.nan, "expected_frauds": 0.0}

    expected_tp = float(scores[flagged].sum())
    expected_total_fraud = float(scores.sum())
    return {
        "alert_rate": float(flagged.mean()),
        "expected_precision": expected_tp / flagged.sum(),
        "expected_recall": expected_tp / expected_total_fraud if expected_total_fraud else np.nan,
        "expected_frauds": expected_total_fraud,
    }


@dataclass
class Baseline:
    """What normal looked like when the model was accepted."""

    created: str
    threshold: float
    score_bins: list          # [edges, proportions]
    alert_rate: float
    expected_precision: float
    feature_bins: dict[str, list] = field(default_factory=dict)

    @classmethod
    def build(cls, scores: np.ndarray, threshold: float,
              features: pd.DataFrame | None = None,
              max_features: int = 20) -> "Baseline":
        perf = estimate_performance(scores, threshold)
        q = np.linspace(0, 1, 11)

        feature_bins = {}
        if features is not None:
            # Only the most-populated columns: a feature that is 95% missing
            # produces a PSI dominated by its missingness rather than its values.
            usable = [c for c in features.columns if c not in TIME_INDEX_FEATURES]
            coverage = features[usable].notna().mean().sort_values(ascending=False)
            for col in coverage.head(max_features).index:
                values = features[col].to_numpy(dtype="float64", na_value=np.nan)
                values = values[np.isfinite(values)]
                if len(values) > 100:
                    edges, proportions = build_bins(values)
                    if edges:
                        feature_bins[col] = [edges, proportions]

        return cls(
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            threshold=float(threshold),
            score_bins=list(build_bins(scores)),
            alert_rate=perf["alert_rate"],
            expected_precision=perf["expected_precision"],
            feature_bins=feature_bins,
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        return cls(**json.loads(path.read_text()))


def check(baseline: Baseline, scores: np.ndarray,
          features: pd.DataFrame | None = None) -> list[DriftFinding]:
    """Compare a period against the baseline and report what moved.

    Ordered by severity so the caller reads the worst first.
    """
    findings: list[DriftFinding] = []
    perf = estimate_performance(scores, baseline.threshold)

    # --- alert rate: the operational one -----------------------------------
    # A moved alert rate is a capacity problem before it is a quality problem:
    # if 5% of traffic now scores above the threshold, the review team is
    # swamped whether or not the model is still correct.
    rate_change = abs(perf["alert_rate"] - baseline.alert_rate) / max(baseline.alert_rate, 1e-9)
    findings.append(DriftFinding(
        "alert_rate",
        Severity.CRITICAL if rate_change > 2 * ALERT_RATE_TOLERANCE
        else Severity.WARN if rate_change > ALERT_RATE_TOLERANCE else Severity.OK,
        baseline.alert_rate, perf["alert_rate"],
        "share of traffic flagged"))

    # --- estimated precision, from calibration only -------------------------
    if np.isfinite(perf["expected_precision"]) and np.isfinite(baseline.expected_precision):
        drop = (baseline.expected_precision - perf["expected_precision"]) / max(baseline.expected_precision, 1e-9)
        findings.append(DriftFinding(
            "expected_precision",
            Severity.CRITICAL if drop > 0.20 else
            Severity.WARN if drop > 0.10 else
            Severity.WATCH if drop > 0.05 else Severity.OK,
            baseline.expected_precision, perf["expected_precision"],
            "label-free estimate; valid only while calibration holds"))

    # --- score distribution --------------------------------------------------
    psi = psi_from_bins(baseline.score_bins[0], baseline.score_bins[1], scores)
    findings.append(DriftFinding(
        "score_psi",
        Severity.CRITICAL if psi > PSI_MAJOR else
        Severity.WARN if psi > PSI_MINOR else Severity.OK,
        0.0, psi, f"population stability index (>{PSI_MINOR} minor, >{PSI_MAJOR} major)"))

    # --- input features ------------------------------------------------------
    if features is not None and baseline.feature_bins:
        drifted = []
        for col, (edges, proportions) in baseline.feature_bins.items():
            if col not in features.columns or col in TIME_INDEX_FEATURES:
                continue
            current = features[col].to_numpy(dtype="float64", na_value=np.nan)
            col_psi = psi_from_bins(edges, proportions, current)
            if col_psi > PSI_MAJOR:
                drifted.append((col, col_psi))

        drifted.sort(key=lambda x: -x[1])
        worst = ", ".join(f"{c} {v:.2f}" for c, v in drifted[:3]) or "none"
        findings.append(DriftFinding(
            "features_drifted",
            Severity.CRITICAL if len(drifted) >= 5 else
            Severity.WARN if len(drifted) >= 2 else
            Severity.WATCH if drifted else Severity.OK,
            0.0, float(len(drifted)),
            f"of {len(baseline.feature_bins)} watched (PSI>{PSI_MAJOR}); worst: {worst}"))

    order = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.WATCH: 2, Severity.OK: 3}
    return sorted(findings, key=lambda f: order[f.severity])


def label_backfill(scores: np.ndarray, labels: np.ndarray,
                   threshold: float) -> dict:
    """Once labels arrive, check whether the label-free estimate was honest.

    CBPE assumes calibration holds. If the feature-to-fraud relationship has
    changed, the estimate can be confidently wrong - and the only way to find
    that out is to compare it against truth when truth turns up.

    A large gap here means the estimate itself can no longer be trusted, which
    is a more serious finding than the drift it failed to detect.
    """
    flagged = scores >= threshold
    estimated = estimate_performance(scores, threshold)
    actual_precision = labels[flagged].mean() if flagged.any() else np.nan
    actual_recall = labels[flagged].sum() / labels.sum() if labels.sum() else np.nan

    gap = abs(estimated["expected_precision"] - actual_precision)
    return {
        "estimated_precision": estimated["expected_precision"],
        "actual_precision": float(actual_precision),
        "actual_recall": float(actual_recall),
        "estimate_error": float(gap),
        "estimate_trustworthy": bool(gap < 0.10),
    }


def format_report(findings: list[DriftFinding]) -> str:
    worst = findings[0].severity if findings else Severity.OK
    lines = [f"DRIFT REPORT - worst severity: {worst.upper()}", "-" * 78]
    lines += [f.describe() for f in findings]
    if worst == Severity.CRITICAL:
        lines += ["", "CRITICAL: retrain, and check label_backfill before trusting",
                  "the estimated numbers above."]
    return "\n".join(lines)
