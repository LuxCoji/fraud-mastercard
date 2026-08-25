"""Experiment C: are the 219 discarded V columns carrying anything?

Deotte's correlation analysis kept 120 of the 339 V columns and dropped the rest
as near-duplicates. But V258 and V257 turned out to be the two strongest
features in our model, so the block clearly matters more than the shortlist
assumed. This trains the same model on all 339 and compares.

The comparison is single-seed, so only a clear gap should be believed.

Run: python src/models/experiment_v_columns.py
"""
from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=Warning)

import config as cfg
from data import loaders
from features.card import CardFeatureBuilder
from models.metrics import format_operating_points, headline, operating_points
from models.train_card import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS
from validation.splits import chronological_split


def run(keep_v, label: str, train_df, test_df, y_train, y_test) -> dict:
    import lightgbm as lgb

    t0 = time.time()
    builder = CardFeatureBuilder(keep_v=keep_v).fit(train_df)
    X_train, X_test = builder.transform(train_df), builder.transform(test_df)

    cut = int(len(X_train) * 0.9)
    model = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X_train.iloc[:cut], y_train[:cut]),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_train.iloc[cut:], y_train[cut:])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    scores = model.predict(X_test, num_iteration=model.best_iteration)

    h = headline(y_test, scores)
    points = operating_points(y_test, scores)
    at_two_pct = points[points.alert_rate == 0.02].iloc[0]

    print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
    print(f"  features        {X_train.shape[1]}")
    print(f"  rounds chosen   {model.best_iteration}")
    print(f"  AUC             {h['auc']:.4f}")
    print(f"  PR-AUC          {h['pr_auc']:.4f}")
    print(f"  Brier           {h['brier']:.5f}")
    print(f"  minutes         {(time.time() - t0) / 60:.1f}")
    print()
    print(format_operating_points(points))

    n_features = X_train.shape[1]
    del X_train, X_test, model
    gc.collect()
    return {
        "variant": label,
        "features": n_features,
        "auc": h["auc"],
        "pr_auc": h["pr_auc"],
        "brier": h["brier"],
        "recall_at_2pct": at_two_pct.recall,
        "genuine_hit_at_2pct": at_two_pct.genuine_disturbed_pct,
    }


def main() -> None:
    df = loaders.load_card()
    split = chronological_split(df, test_fraction=0.20)
    train_df, test_df = df.iloc[split.train_idx], df.iloc[split.test_idx]
    y_train = train_df["isFraud"].values
    y_test = test_df["isFraud"].values
    del df
    gc.collect()

    rows = [
        run("curated", "CURATED - 120 V columns (current)", train_df, test_df, y_train, y_test),
        run("all", "ALL - 339 V columns", train_df, test_df, y_train, y_test),
    ]

    out = pd.DataFrame(rows)
    print(f"\n{'=' * 62}\nCOMPARISON\n{'=' * 62}")
    print(out.to_string(index=False))
    out.to_csv(cfg.REPORTS / "step2_v_column_experiment.csv", index=False)

    delta = (rows[1]["pr_auc"] - rows[0]["pr_auc"]) / rows[0]["pr_auc"]
    print(f"\nPR-AUC change from keeping all V columns: {delta:+.2%}")
    print("Single seed - anything under about 1% is noise, not a result.")


if __name__ == "__main__":
    main()
