"""Train the transfer rail (PaySim).

Same shape as the card rail: a logistic floor first, then LightGBM, then the
operating points that actually drive the policy. Nothing is rebalanced, and the
split is chronological.

One difference worth stating up front. The router drops PAYMENT, CASH_IN and
DEBIT before anything is scored, because those three types contain zero fraud
across all 6,362,620 rows. That is not a modelling decision - it is a fact about
the rail, and encoding it as a rule rather than hoping a model rediscovers it
removes 56.5% of traffic from the scoring path at no risk.

Run: python src/models/train_transfer.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data import loaders
from features.transfer import SCOREABLE_TYPES, build_features, route
from models.metrics import format_operating_points, headline, operating_points
from models.train_card import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS
from validation.splits import chronological_split


def train_logistic(X_train, y_train, X_test):
    """The floor. No class weighting - rebalancing destroys calibration."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, n_jobs=-1),
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def train_lightgbm(X_train, y_train):
    """LightGBM with the card rail's parameters, tree count re-fit here."""
    import lightgbm as lgb

    cut = int(len(X_train) * 0.9)
    model = lgb.train(
        dict(LGB_PARAMS, seed=cfg.RANDOM_SEED),
        lgb.Dataset(X_train.iloc[:cut], y_train[:cut]),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_train.iloc[cut:], y_train[cut:])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(200)],
    )
    print(f"  early stopping chose {model.best_iteration} rounds", flush=True)
    return model


def report(name: str, y_test, scores) -> dict:
    h = headline(y_test, scores)
    state = "OK" if h["brier"] <= h["brier_baseline"] else "WORSE THAN BASELINE"
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  PR-AUC  {h['pr_auc']:.4f}    (base rate {h['base_rate']:.4%})")
    print(f"  AUC     {h['auc']:.4f}")
    print(f"  Brier   {h['brier']:.6f}  vs baseline {h['brier_baseline']:.6f}   {state}")
    print()
    print(format_operating_points(operating_points(y_test, scores)))
    return h


def main() -> None:
    raw = loaders.load_transfer()
    total_rows, total_fraud = len(raw), int(raw.isFraud.sum())

    # --- router -------------------------------------------------------------
    scoreable = route(raw)
    dropped = total_rows - len(scoreable)
    dropped_fraud = total_fraud - int(scoreable.isFraud.sum())

    print(f"{'=' * 62}\nROUTER\n{'=' * 62}")
    print(f"  all traffic       {total_rows:>10,} rows  {total_fraud:>6,} fraud")
    print(f"  dropped           {dropped:>10,} rows  {dropped_fraud:>6,} fraud "
          f"({dropped / total_rows:.1%} of traffic)")
    print(f"  scored            {len(scoreable):>10,} rows  "
          f"{int(scoreable.isFraud.sum()):>6,} fraud")
    print(f"  types scored      {', '.join(SCOREABLE_TYPES)}")
    print(f"  fraud rate        {raw.isFraud.mean():.4%} -> "
          f"{scoreable.isFraud.mean():.4%} after routing")
    if dropped_fraud:
        print(f"  WARNING: routing discarded {dropped_fraud} fraud cases")
    del raw
    gc.collect()

    # --- split --------------------------------------------------------------
    split = chronological_split(scoreable, time_col="step", test_fraction=0.20)
    train_df = scoreable.iloc[split.train_idx]
    test_df = scoreable.iloc[split.test_idx]
    y_train = train_df["isFraud"].values
    y_test = test_df["isFraud"].values

    X_train = build_features(train_df)
    X_test = build_features(test_df)
    del scoreable, train_df, test_df
    gc.collect()

    print(f"\n{'=' * 62}\nSPLIT\n{'=' * 62}")
    print(f"  train  {len(X_train):>9,} rows  {int(y_train.sum()):>6,} fraud "
          f"({y_train.mean():.4%})")
    print(f"  test   {len(X_test):>9,} rows  {int(y_test.sum()):>6,} fraud "
          f"({y_test.mean():.4%})")
    print(f"  features {X_train.shape[1]}: {', '.join(X_train.columns)}", flush=True)

    results = {}

    t0 = time.time()
    scores_lr = train_logistic(X_train, y_train, X_test)
    print(f"\nlogistic trained in {time.time() - t0:.0f}s", flush=True)
    results["logistic"] = report("RUNG 1 - Logistic regression (the floor)",
                                 y_test, scores_lr)

    t0 = time.time()
    model = train_lightgbm(X_train, y_train)
    scores_lgb = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"\nlightgbm trained in {time.time() - t0:.0f}s", flush=True)
    results["lightgbm"] = report("RUNG 2 - LightGBM", y_test, scores_lgb)

    np.save(cfg.INTERIM / "transfer_scores_lightgbm.npy", scores_lgb)
    np.save(cfg.INTERIM / "transfer_y_test.npy", y_test)
    model.save_model(str(cfg.ARTIFACTS / "transfer_lightgbm.txt"))
    operating_points(y_test, scores_lgb).to_csv(
        cfg.REPORTS / "step3_transfer_operating_points.csv", index=False)

    lift = results["lightgbm"]["pr_auc"] / results["logistic"]["pr_auc"]
    print(f"\n{'=' * 62}")
    print(f"LightGBM PR-AUC is {lift:.2f}x the logistic floor.")

    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(cfg.REPORTS / "step3_transfer_feature_importance.csv", index=False)
    print("\nFeatures by gain:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
