"""Train the wire rail on HI-Medium, on a GPU, and compare against HI-Small.

HI-Small gave a working model but an unreliable one: PR-AUC +-0.029 across three
seeds (about 19% relative), early stopping anywhere between 75 and 588 rounds,
and a headline verdict that passed its own test by 0.0002. The stated cause was
the positive count - 3,380 laundering cases in four million rows means the model
learns from a very small sample and which subset bagging draws matters enormously.

HI-Medium is the test of that explanation:

    HI-Small    5,078,345 rows      370 labelled rings
    HI-Medium  31,898,239 rows    2,756 labelled rings

If the instability was a sample-size problem, the seed spread should shrink
sharply. If it does not, the cause is something else and HI-Large will not help
either - which matters, because IBM's own results show gradient boosting
*degrading* on HI-Large (XGBoost 65.7% on Medium, 42.7% on Large).

Designed to run on Colab: 13.6 GB RAM and a T4. XGBoost uses `device="cuda"`,
which it genuinely exploits; LightGBM is left on CPU since its GPU support is
marginal.

Run on Colab via:
    colab exec --session <id> < this file
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/content/proj/src")
if PROJECT.exists():
    sys.path.insert(0, str(PROJECT))
    DATA = Path("/content/data/HI-Medium_Trans.csv")
    OUT = Path("/content/out")
else:                                   # local fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as _cfg
    DATA = _cfg.RAW / "HI-Medium_Trans.csv"
    OUT = _cfg.REPORTS
OUT.mkdir(parents=True, exist_ok=True)

from features.wire import LABEL, WireFeatureBuilder, sort_by_time
from models.metrics import format_operating_points, headline, operating_points

SEEDS = [42, 7, 2024]
FIT_FRACTION = 0.85
EARLY_STOP_FRACTION = 0.075
MAX_ROUNDS = 3000
EARLY_STOPPING = 100

# Narrow dtypes at read time. 31.9M rows as float64 would not fit in 13.6 GB.
DTYPES = {
    "From Bank": "int32", "To Bank": "int32",
    "Account": "category", "Account.1": "category",
    "Amount Received": "float32", "Amount Paid": "float32",
    "Receiving Currency": "category", "Payment Currency": "category",
    "Payment Format": "category", "Is Laundering": "int8",
}


def load() -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(DATA, dtype=DTYPES)
    print(f"loaded {len(df):,} rows in {(time.time() - t0) / 60:.1f} min, "
          f"{df.memory_usage(deep=True).sum() / 1e9:.1f} GB", flush=True)
    # The account columns are read as category to save memory but the feature
    # builder groups on them, which is far slower on categoricals.
    for col in ("Account", "Account.1"):
        df[col] = df[col].astype(str)
    return df


def fit_xgboost(X_fit, y_fit, X_stop, y_stop, X_test, seed: int):
    import xgboost as xgb
    params = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "max_depth": 8, "learning_rate": 0.02,
        "subsample": 0.8, "colsample_bytree": 0.4,
        "tree_method": "hist", "device": "cuda", "seed": seed,
    }
    dtrain = xgb.QuantileDMatrix(X_fit, label=y_fit, missing=np.nan)
    dvalid = xgb.QuantileDMatrix(X_stop, label=y_stop, missing=np.nan, ref=dtrain)
    booster = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                        evals=[(dvalid, "valid")],
                        early_stopping_rounds=EARLY_STOPPING,
                        verbose_eval=250)
    scores = booster.predict(xgb.DMatrix(X_test, missing=np.nan),
                             iteration_range=(0, booster.best_iteration + 1))
    return scores, booster.best_iteration


def main() -> None:
    df = sort_by_time(load())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")

    split = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    y_train, y_test = train_df[LABEL].values, test_df[LABEL].values

    n = len(train_df)
    fit_end = int(n * FIT_FRACTION)
    stop_end = int(n * (FIT_FRACTION + EARLY_STOP_FRACTION))

    print(f"\nfit        {fit_end:>10,} rows  "
          f"{int(y_train[:fit_end].sum()):>6,} laundering")
    print(f"early stop {stop_end - fit_end:>10,} rows  "
          f"{int(y_train[fit_end:stop_end].sum()):>6,} laundering")
    print(f"test       {len(test_df):>10,} rows  {int(y_test.sum()):>6,} laundering "
          f"({y_test.mean():.4%})\n", flush=True)

    t0 = time.time()
    builder = WireFeatureBuilder(with_patterns=True).fit(train_df.iloc[:fit_end])
    X_fit = builder.transform(train_df.iloc[:fit_end])
    X_stop = builder.transform(train_df.iloc[fit_end:stop_end])
    X_test = builder.transform(test_df)
    y_fit, y_stop = y_train[:fit_end], y_train[fit_end:stop_end]
    print(f"features built in {(time.time() - t0) / 60:.1f} min: "
          f"{X_fit.shape[1]} columns\n", flush=True)

    del df, train_df, test_df
    gc.collect()

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        scores, rounds = fit_xgboost(X_fit, y_fit, X_stop, y_stop, X_test, seed)
        h = headline(y_test, scores)
        points = operating_points(y_test, scores)
        at_one = points[points.alert_rate == 0.01].iloc[0]
        rows.append({"seed": seed, "pr_auc": h["pr_auc"], "auc": h["auc"],
                     "brier": h["brier"], "brier_baseline": h["brier_baseline"],
                     "recall_at_1pct": at_one.recall,
                     "precision_at_1pct": at_one.precision, "rounds": rounds})
        print(f"\nseed {seed}: PR-AUC {h['pr_auc']:.4f}  "
              f"recall@1% {at_one.recall:.2%}  {rounds} rounds  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        np.save(OUT / f"wire_medium_scores_{seed}.npy", scores)
        pd.DataFrame(rows).to_csv(OUT / "wire_medium_seeds.csv", index=False)
        gc.collect()

    out = pd.DataFrame(rows)
    print(f"\n{'=' * 66}\nHI-MEDIUM, THREE SEEDS\n{'=' * 66}")
    print(out.to_string(index=False))

    pr = out.pr_auc.values
    rc = out.recall_at_1pct.values
    print(f"\n  PR-AUC     {pr.mean():.4f} +-{pr.std(ddof=1):.4f}  "
          f"({pr.std(ddof=1) / pr.mean():.1%} relative)")
    print(f"  recall@1%  {rc.mean():.2%} +-{rc.std(ddof=1):.2%}")
    print(f"  rounds     {list(out['rounds'])}")

    print(f"\n{'=' * 66}\nAGAINST HI-SMALL\n{'=' * 66}")
    small = {"pr_auc_mean": 0.1508, "pr_auc_std": 0.0292,
             "recall_mean": 0.5418, "rounds": [399, 92, 153]}
    print(f"  PR-AUC        {small['pr_auc_mean']:.4f} +-{small['pr_auc_std']:.4f} "
          f"-> {pr.mean():.4f} +-{pr.std(ddof=1):.4f}")
    print(f"  spread        {small['pr_auc_std'] / small['pr_auc_mean']:.1%} "
          f"-> {pr.std(ddof=1) / pr.mean():.1%}")
    print(f"  recall@1%     {small['recall_mean']:.2%} -> {rc.mean():.2%}")
    print(f"  rounds        {small['rounds']} -> {list(out['rounds'])}")
    print("\n  The spread is the number that matters: if it fell sharply, the")
    print("  HI-Small instability was a sample-size problem. If not, more data")
    print("  is not the fix and HI-Large will not help either.")

    print("\n" + json.dumps({"pr_auc_mean": float(pr.mean()),
                             "pr_auc_std": float(pr.std(ddof=1)),
                             "recall_mean": float(rc.mean()),
                             "rounds": [int(r) for r in out["rounds"]]}))
    print(format_operating_points(operating_points(
        y_test, np.load(OUT / f"wire_medium_scores_{SEEDS[0]}.npy"))))


if __name__ == "__main__":
    main()
