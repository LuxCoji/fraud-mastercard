"""Build the shippable card-rail model: train, calibrate, evaluate, save.

Experiment D settled that a LightGBM + XGBoost blend beats either alone
(PR-AUC 0.5948 vs 0.5865, winning all three seeds). It left one problem: the
blend was rank-averaged, which throws calibration away. Brier came out at
0.2985 against a base-rate baseline of 0.0332, so the scores could rank
transactions but could not be read as probabilities - and every policy
threshold is a probability.

This fixes that with a three-way chronological split:

    fit          first 85% of the training period    - grows the trees
    early stop   next 7.5%                           - picks the tree count
    calibrate    last 7.5%                           - maps scores to probabilities
    test         the held-out final 20% of all data  - untouched until the end

The calibration slice is deliberately the most recent training data, closest in
time to the period the model will actually score.

Two blending methods are compared rather than assumed:

    A  rank-average the two models, then calibrate the blend
    B  calibrate each model separately, then average the probabilities

A preserves the blend's ranking exactly, because isotonic regression is
monotonic - so PR-AUC and the operating points cannot move, and only Brier
improves. B may reorder slightly. Both are measured.

Run: python src/models/build_card_model.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from models.experiment_ensemble import fit_lightgbm, fit_xgboost, to_rank
from models.metrics import format_operating_points, headline, operating_points

SEED = cfg.RANDOM_SEED
FIT_FRACTION = 0.85
EARLY_STOP_FRACTION = 0.075


def isotonic(x_cal: np.ndarray, y_cal: np.ndarray):
    """Fit a step-shaped monotonic score -> probability map.

    Flexible, but it is a *step* function: it collapses ranges of input into a
    single output value. Measured on this data it turned 118,108 distinct scores
    into 153, tying 99.87% of rows, which cost 0.027 PR-AUC. Kept here only for
    comparison - see `platt`.
    """
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(x_cal, y_cal)
    return model


def platt(x_cal: np.ndarray, y_cal: np.ndarray):
    """Fit a sigmoid score -> probability map.

    A sigmoid is *strictly* increasing, so no two distinct scores collapse onto
    the same probability. Ranking is preserved exactly, which means PR-AUC and
    every operating point are untouched and only the probability scale changes.

    Less flexible than isotonic - it cannot correct an arbitrarily shaped
    miscalibration - but on this rail that flexibility was not worth the ties.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(x_cal.reshape(-1, 1), y_cal)

    class _Sigmoid:
        def predict(self, x):
            return model.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]

    return _Sigmoid()


def report(name: str, y_true, scores) -> dict:
    h = headline(y_true, scores)
    calibrated = "OK" if h["brier"] <= h["brier_baseline"] else "WORSE THAN BASELINE"
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  PR-AUC  {h['pr_auc']:.4f}")
    print(f"  AUC     {h['auc']:.4f}")
    print(f"  Brier   {h['brier']:.5f}  vs baseline {h['brier_baseline']:.5f}   {calibrated}")
    print()
    print(format_operating_points(operating_points(y_true, scores)))
    return h


def main() -> None:
    X_train = pd.read_parquet(cfg.INTERIM / "card_X_train.parquet")
    X_test = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")
    y_train = np.load(cfg.INTERIM / "card_y_train.npy")
    y_test = np.load(cfg.INTERIM / "card_y_test.npy")

    n = len(X_train)
    fit_end = int(n * FIT_FRACTION)
    stop_end = int(n * (FIT_FRACTION + EARLY_STOP_FRACTION))

    X_fit, y_fit = X_train.iloc[:fit_end], y_train[:fit_end]
    X_stop, y_stop = X_train.iloc[fit_end:stop_end], y_train[fit_end:stop_end]
    X_cal, y_cal = X_train.iloc[stop_end:], y_train[stop_end:]

    print(f"fit        {len(X_fit):>8,} rows  {int(y_fit.sum()):>6,} fraud")
    print(f"early stop {len(X_stop):>8,} rows  {int(y_stop.sum()):>6,} fraud")
    print(f"calibrate  {len(X_cal):>8,} rows  {int(y_cal.sum()):>6,} fraud")
    print(f"test       {len(X_test):>8,} rows  {int(y_test.sum()):>6,} fraud\n", flush=True)

    if y_cal.sum() < 500:
        print("WARNING: calibration slice has few positives; isotonic may overfit.",
              flush=True)

    # Predict on calibration and test in one pass per model, so each model is
    # trained exactly once.
    X_both = pd.concat([X_cal, X_test], axis=0)
    split_at = len(X_cal)

    raw = {}
    for name, build in (("lightgbm", fit_lightgbm), ("xgboost", fit_xgboost)):
        # Cached so that comparing calibration methods never costs a retrain -
        # the first pass at this had to retrain to test a second method.
        cache = cfg.INTERIM / f"card_final_raw_{name}.npy"
        if cache.exists():
            print(f"--- {name}: reusing saved scores ---", flush=True)
            both = np.load(cache)
        else:
            t0 = time.time()
            print(f"--- {name} ---", flush=True)
            both = build(X_fit, y_fit, X_stop, y_stop, X_both, SEED)
            np.save(cache, both)
            print(f"    trained in {(time.time() - t0) / 60:.1f} min", flush=True)
        raw[name] = {"cal": both[:split_at], "test": both[split_at:]}
        gc.collect()

    results = {}

    # --- baselines, uncalibrated ------------------------------------------
    for name in ("lightgbm", "xgboost"):
        results[name] = report(f"{name} (raw)", y_test, raw[name]["test"])

    # --- method A: rank-average, then calibrate ---------------------------
    rank_cal = np.mean([to_rank(raw[m]["cal"]) for m in raw], axis=0)
    rank_test = np.mean([to_rank(raw[m]["test"]) for m in raw], axis=0)
    results["blend_rank_raw"] = report("BLEND A - rank-averaged, uncalibrated",
                                       y_test, rank_test)

    a_test = isotonic(rank_cal, y_cal).predict(rank_test)
    results["blend_isotonic"] = report(
        "BLEND A - rank-averaged, isotonic", y_test, a_test)

    # --- method B: calibrate each model, then average ---------------------
    b_test = np.mean(
        [isotonic(raw[n]["cal"], y_cal).predict(raw[n]["test"]) for n in raw],
        axis=0)
    results["blend_isotonic_each"] = report(
        "BLEND B - each model isotonic, then averaged", y_test, b_test)

    # --- method C: rank-average, then Platt -------------------------------
    c_test = platt(rank_cal, y_cal).predict(rank_test)
    results["blend_platt"] = report(
        "BLEND C - rank-averaged, Platt (strictly monotonic)", y_test, c_test)

    # --- summary ----------------------------------------------------------
    candidates = {"blend_isotonic": a_test, "blend_isotonic_each": b_test,
                  "blend_platt": c_test}

    table = pd.DataFrame(results).T[["pr_auc", "auc", "brier"]]
    table["brier_baseline"] = headline(y_test, c_test)["brier_baseline"]
    table["calibrated"] = table.brier <= table.brier_baseline
    table["distinct_scores"] = [
        len(np.unique(candidates[k])) if k in candidates else np.nan
        for k in table.index
    ]
    print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
    print(table.to_string())
    table.to_csv(cfg.REPORTS / "step2_card_calibration.csv")

    # Pick the calibrated candidate that keeps the most ranking. Brier decides
    # whether a method is usable at all; among the usable ones, PR-AUC decides,
    # because the policy reads the ranking and only needs the probability to be
    # honest - not to be the single lowest Brier available.
    usable = {k: v for k, v in results.items()
              if k in candidates and v["brier"] <= v["brier_baseline"]}
    best = max(usable, key=lambda k: usable[k]["pr_auc"])
    print(f"\nAll calibrated: {sorted(usable)}")
    print(f"Best ranking among them: {best} (PR-AUC {usable[best]['pr_auc']:.4f})")

    chosen = candidates[best]
    np.save(cfg.INTERIM / "card_scores_final.npy", chosen)
    (cfg.ARTIFACTS / "card_model_card.json").write_text(json.dumps({
        "rail": "card",
        "source": "ieee_cis",
        "models": ["lightgbm", "xgboost"],
        "blend": best,
        "seed": SEED,
        "split": {"fit": len(X_fit), "early_stop": len(X_stop),
                  "calibrate": len(X_cal), "test": len(X_test)},
        "metrics": {k: {m: float(v[m]) for m in ("pr_auc", "auc", "brier")}
                    for k, v in results.items()},
    }, indent=2))
    print(f"saved -> {cfg.ARTIFACTS / 'card_model_card.json'}")


if __name__ == "__main__":
    main()
