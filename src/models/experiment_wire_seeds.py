"""Is XGBoost really better than LightGBM on the wire rail, or was that one seed?

A single run put XGBoost at PR-AUC 0.1831 against LightGBM's 0.1035 - a 77% gap,
and the reverse of the card rail where the two were close and blending won. The
gap is large enough to look real, but the card rail measured XGBoost's seed
spread at +-0.0170 against LightGBM's +-0.0024, so XGBoost is exactly the model
whose single run is least worth trusting.

Decision rule, fixed before results: adopt XGBoost as the wire model only if it
beats LightGBM by more than the combined seed spread AND wins on every seed.

Features are deterministic, so they are built once and reused across seeds.

Run: python src/models/experiment_wire_seeds.py
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
from models.metrics import headline, operating_points
from models.train_card import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS

SEEDS = [42, 7, 2024]
FIT_FRACTION = 0.85
EARLY_STOP_FRACTION = 0.075


def fit_lightgbm(X_fit, y_fit, X_stop, y_stop, X_pred, seed):
    import lightgbm as lgb
    model = lgb.train(
        dict(LGB_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed),
        lgb.Dataset(X_fit, y_fit),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_stop, y_stop)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return model.predict(X_pred, num_iteration=model.best_iteration), model.best_iteration


def fit_xgboost(X_fit, y_fit, X_stop, y_stop, X_pred, seed):
    import xgboost as xgb
    params = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "max_depth": 8, "learning_rate": 0.02,
        "subsample": 0.8, "colsample_bytree": 0.4,
        "tree_method": "hist", "nthread": -1, "seed": seed,
    }
    dtrain = xgb.QuantileDMatrix(X_fit, label=y_fit, missing=np.nan)
    dvalid = xgb.QuantileDMatrix(X_stop, label=y_stop, missing=np.nan, ref=dtrain)
    booster = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                        evals=[(dvalid, "valid")],
                        early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
    scores = booster.predict(xgb.DMatrix(X_pred, missing=np.nan),
                             iteration_range=(0, booster.best_iteration + 1))
    return scores, booster.best_iteration


BUILDERS = {"lightgbm": fit_lightgbm, "xgboost": fit_xgboost}


def main() -> None:
    df = sort_by_time(loaders.load_wire())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")
    split = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    y_train, y_test = train_df[LABEL].values, test_df[LABEL].values
    del df
    gc.collect()

    n = len(train_df)
    fit_end = int(n * FIT_FRACTION)
    stop_end = int(n * (FIT_FRACTION + EARLY_STOP_FRACTION))

    t0 = time.time()
    builder = WireFeatureBuilder(with_patterns=True).fit(train_df.iloc[:fit_end])
    X_fit = builder.transform(train_df.iloc[:fit_end])
    X_stop = builder.transform(train_df.iloc[fit_end:stop_end])
    X_test = builder.transform(test_df)
    y_fit, y_stop = y_train[:fit_end], y_train[fit_end:stop_end]
    print(f"features built once in {(time.time() - t0) / 60:.1f} min: "
          f"{X_fit.shape[1]} columns")
    print(f"fit {len(X_fit):,}  early stop {len(X_stop):,}  "
          f"test {len(X_test):,} ({int(y_test.sum()):,} laundering)\n", flush=True)
    del train_df, test_df
    gc.collect()

    rows = []
    for seed in SEEDS:
        for name, fit in BUILDERS.items():
            cache = cfg.INTERIM / f"wire_seed_{name}_{seed}.npy"
            if cache.exists():
                scores, rounds = np.load(cache), -1
                print(f"seed {seed:>5} | {name:<9} reusing saved scores", flush=True)
            else:
                t0 = time.time()
                scores, rounds = fit(X_fit, y_fit, X_stop, y_stop, X_test, seed)
                np.save(cache, scores)
                print(f"seed {seed:>5} | {name:<9} trained in "
                      f"{(time.time() - t0) / 60:.1f} min ({rounds} rounds)", flush=True)

            h = headline(y_test, scores)
            points = operating_points(y_test, scores)
            at_one = points[points.alert_rate == 0.01].iloc[0]
            rows.append({"seed": seed, "model": name, "pr_auc": h["pr_auc"],
                         "auc": h["auc"], "brier": h["brier"],
                         "brier_baseline": h["brier_baseline"],
                         "recall_at_1pct": at_one.recall,
                         "precision_at_1pct": at_one.precision, "rounds": rounds})
            pd.DataFrame(rows).to_csv(
                cfg.REPORTS / "step4_wire_seed_experiment.csv", index=False)
            print(f"              PR-AUC {h['pr_auc']:.4f}  "
                  f"recall@1% {at_one.recall:.2%}  "
                  f"Brier {h['brier']:.6f}", flush=True)
            gc.collect()

    out = pd.DataFrame(rows)
    print(f"\n{'=' * 70}\nPER SEED\n{'=' * 70}")
    print(out.pivot(index="seed", columns="model", values="pr_auc").to_string())

    summary = out.groupby("model").agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        recall_mean=("recall_at_1pct", "mean"), brier_mean=("brier", "mean"),
    )
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(summary.to_string())

    lgb_scores = out[out.model == "lightgbm"].sort_values("seed").pr_auc.values
    xgb_scores = out[out.model == "xgboost"].sort_values("seed").pr_auc.values
    gap = xgb_scores.mean() - lgb_scores.mean()
    spread = np.sqrt(lgb_scores.std(ddof=1) ** 2 + xgb_scores.std(ddof=1) ** 2)
    every = bool((xgb_scores > lgb_scores).all())

    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    print(f"  mean gap        {gap:+.4f}  ({gap / lgb_scores.mean():+.1%})")
    print(f"  combined spread {spread:.4f}")
    print(f"  xgboost wins every seed: {every}")
    print(f"\n  -> {'ADOPT XGBOOST' if gap > spread and every else 'NOT PROVEN'}")


if __name__ == "__main__":
    main()
