"""Train the wire rail (IBM AML).

Same discipline as the card rail: logistic floor first, chronological split, no
rebalancing, and the operating points rather than accuracy.

Two things are specific to this rail.

First, the rows must be sorted by time before splitting. IBM AML's timestamps
are not monotonic in the file - confirmed in step 1 - and splitting an unsorted
frame trains the model on its own future.

Second, the features are account-level rather than row-level, because laundering
is a topology. The 370 labelled rings are CYCLE, FAN-IN, FAN-OUT and five other
shapes, none of which exists inside a single transaction.

Standing caveat: the source is synthetic. It was checked for the artefact that
made PaySim unusable and is clean, but clean is not real.

Run: python src/models/train_wire.py
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
from features.wire import LABEL, WireFeatureBuilder, sort_by_time
from models.metrics import format_operating_points, headline, operating_points
from models.train_card import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS


def train_logistic(X_train, y_train, X_test):
    """The floor. No class weighting."""
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
    df = loaders.load_wire()
    print(f"loaded {len(df):,} rows, {int(df[LABEL].sum()):,} laundering "
          f"({df[LABEL].mean():.4%})", flush=True)

    # Sorting is not optional here - see the module docstring.
    df = sort_by_time(df)
    cut = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    y_train = train_df[LABEL].values
    y_test = test_df[LABEL].values

    print(f"\n{'=' * 62}\nSPLIT (chronological, after sorting)\n{'=' * 62}")
    print(f"  train {len(train_df):>9,} rows  {int(y_train.sum()):>5,} laundering "
          f"({y_train.mean():.4%})")
    print(f"  test  {len(test_df):>9,} rows  {int(y_test.sum()):>5,} laundering "
          f"({y_test.mean():.4%})")
    print(f"  train period {train_df.Timestamp.iloc[0]} .. {train_df.Timestamp.iloc[-1]}")
    print(f"  test  period {test_df.Timestamp.iloc[0]} .. {test_df.Timestamp.iloc[-1]}",
          flush=True)

    if y_test.sum() < 200:
        print("\n  WARNING: few laundering cases in test; metrics will be noisy.")

    t0 = time.time()
    builder = WireFeatureBuilder().fit(train_df)      # train only
    X_train = builder.transform(train_df)
    X_test = builder.transform(test_df)
    print(f"\nfeatures built in {time.time() - t0:.0f}s: "
          f"{X_train.shape[1]} columns", flush=True)
    print("  " + ", ".join(X_train.columns))

    unseen = X_test["sent_txns"].isna().mean()
    print(f"  test rows whose sender was unseen in training: {unseen:.1%}")

    del df, train_df, test_df
    gc.collect()

    results = {}

    t0 = time.time()
    scores_lr = train_logistic(X_train, y_train, X_test)
    print(f"\nlogistic trained in {time.time() - t0:.0f}s", flush=True)
    results["logistic"] = report("RUNG 1 - Logistic regression (the floor)",
                                 y_test, scores_lr)

    import lightgbm as lgb
    t0 = time.time()
    fit_cut = int(len(X_train) * 0.9)
    model = lgb.train(
        dict(LGB_PARAMS, seed=cfg.RANDOM_SEED),
        lgb.Dataset(X_train.iloc[:fit_cut], y_train[:fit_cut]),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_train.iloc[fit_cut:], y_train[fit_cut:])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(200)],
    )
    print(f"  early stopping chose {model.best_iteration} rounds")
    scores_lgb = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"lightgbm trained in {time.time() - t0:.0f}s", flush=True)
    results["lightgbm"] = report("RUNG 2 - LightGBM on account-graph features",
                                 y_test, scores_lgb)

    np.save(cfg.INTERIM / "wire_scores_lightgbm.npy", scores_lgb)
    np.save(cfg.INTERIM / "wire_y_test.npy", y_test)
    model.save_model(str(cfg.ARTIFACTS / "wire_lightgbm.txt"))
    operating_points(y_test, scores_lgb).to_csv(
        cfg.REPORTS / "step4_wire_operating_points.csv", index=False)

    lift = results["lightgbm"]["pr_auc"] / max(results["logistic"]["pr_auc"], 1e-9)
    print(f"\n{'=' * 62}")
    print(f"LightGBM PR-AUC is {lift:.2f}x the logistic floor.")

    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(cfg.REPORTS / "step4_wire_feature_importance.csv", index=False)
    print("\nTop features by gain:")
    print(importance.head(15).to_string(index=False))

    graph_cols = [c for c in X_train.columns
                  if c.startswith(("sent_", "recv_")) or "passthrough" in c]
    graph_share = importance[importance.feature.isin(graph_cols)].gain.sum() / importance.gain.sum()
    print(f"\nAccount-graph features carry {graph_share:.1%} of total gain.")
    print("If that were near zero, the graph framing would not be earning its keep.")


if __name__ == "__main__":
    main()
