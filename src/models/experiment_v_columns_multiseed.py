"""Experiment C, settled properly: 120 V columns vs 339, over three seeds.

The single-seed run gave +1.22% PR-AUC for keeping all 339 columns, which sits
right on the noise floor. This repeats both variants across three seeds so the
question can be answered rather than guessed at.

Feature building is deterministic, so it runs once per variant; only the
LightGBM seed changes between runs. The decision rule is fixed before the
results are seen:

    the gap counts as real only if the mean improvement exceeds the spread of
    the seeds, and holds the same direction in all three.

Run: python src/models/experiment_v_columns_multiseed.py
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
from models.metrics import headline, operating_points
from models.train_card import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS
from validation.splits import chronological_split

SEEDS = [42, 7, 2024]
VARIANTS = [("curated", "120 V columns"), ("all", "339 V columns")]


def train_one(X_train, y_train, X_test, y_test, seed: int) -> dict:
    import lightgbm as lgb

    params = dict(LGB_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    cut = int(len(X_train) * 0.9)
    model = lgb.train(
        params,
        lgb.Dataset(X_train.iloc[:cut], y_train[:cut]),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_train.iloc[cut:], y_train[cut:])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    scores = model.predict(X_test, num_iteration=model.best_iteration)
    h = headline(y_test, scores)
    at_two = operating_points(y_test, scores)
    at_two = at_two[at_two.alert_rate == 0.02].iloc[0]

    rounds = model.best_iteration
    del model
    gc.collect()
    return {
        "auc": h["auc"],
        "pr_auc": h["pr_auc"],
        "brier": h["brier"],
        "recall_at_2pct": at_two.recall,
        "genuine_hit_at_2pct": at_two.genuine_disturbed_pct,
        "rounds": rounds,
    }


def main() -> None:
    df = loaders.load_card()
    split = chronological_split(df, test_fraction=0.20)
    train_df, test_df = df.iloc[split.train_idx], df.iloc[split.test_idx]
    y_train, y_test = train_df["isFraud"].values, test_df["isFraud"].values
    del df
    gc.collect()

    rows = []
    for keep_v, label in VARIANTS:
        t0 = time.time()
        builder = CardFeatureBuilder(keep_v=keep_v).fit(train_df)
        X_train, X_test = builder.transform(train_df), builder.transform(test_df)
        print(f"\n{label}: {X_train.shape[1]} features, "
              f"built in {time.time() - t0:.0f}s", flush=True)

        for seed in SEEDS:
            t0 = time.time()
            result = train_one(X_train, y_train, X_test, y_test, seed)
            result.update(variant=label, seed=seed, features=X_train.shape[1])
            rows.append(result)
            print(f"  seed {seed:>5}: PR-AUC {result['pr_auc']:.4f}  "
                  f"AUC {result['auc']:.4f}  recall@2% {result['recall_at_2pct']:.3%}  "
                  f"({result['rounds']} rounds, {(time.time() - t0) / 60:.1f} min)",
                  flush=True)

        del X_train, X_test, builder
        gc.collect()

    out = pd.DataFrame(rows)
    out.to_csv(cfg.REPORTS / "step2_v_column_multiseed.csv", index=False)

    print(f"\n{'=' * 70}\nPER-SEED RESULTS\n{'=' * 70}")
    print(out[["variant", "seed", "features", "pr_auc", "auc",
               "recall_at_2pct", "rounds"]].to_string(index=False))

    summary = out.groupby("variant").agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        auc_mean=("auc", "mean"),
        recall_mean=("recall_at_2pct", "mean"), recall_std=("recall_at_2pct", "std"),
    )
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(summary.to_string())

    curated = out[out.variant == "120 V columns"].pr_auc.values
    every = out[out.variant == "339 V columns"].pr_auc.values
    gap = every.mean() - curated.mean()
    spread = np.sqrt(curated.std(ddof=1) ** 2 + every.std(ddof=1) ** 2)
    same_direction = bool((every > curated).all())

    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    print(f"  mean gap in PR-AUC      {gap:+.4f}  ({gap / curated.mean():+.2%})")
    print(f"  combined seed spread    {spread:.4f}")
    print(f"  all three seeds agree   {same_direction}")
    real = gap > spread and same_direction
    print(f"\n  -> {'REAL - adopt 339 columns' if real else 'NOT PROVEN - keep 120 columns'}")
    if not real and same_direction:
        print("     (direction is consistent, but the gap is inside seed noise)")


if __name__ == "__main__":
    main()
