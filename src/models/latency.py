"""How long does a decision actually take?

A card authorisation has roughly 100 ms end to end, of which fraud scoring gets
a slice. Nothing in this project has been timed yet, so the models might be
comfortably fast or might not fit at all - and that changes the design, not just
the tuning.

Three things are measured, because they fail differently:

  **Model inference** - the part everyone measures, and usually the smallest.
  **Feature building** - the part that usually dominates, because the card rail
  computes 292 features per transaction including per-UID aggregates.
  **The whole path** - what the caller actually waits for.

Everything is measured at **batch 1**. A payment arrives alone; throughput
numbers from batched scoring say nothing about what one customer waits for.

Reported at p99, not the mean. The mean hides exactly the tail that breaks an
SLA - if one authorisation in a hundred takes 400 ms, that is a production
incident, and a 12 ms average will not show it.

Run: python src/models/latency.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

N_SAMPLES = 500          # enough for a stable p99 without taking all day
WARMUP = 50              # first calls pay import and allocation costs
CARD_BUDGET_MS = 100     # the authorisation window the scorer must fit inside


def percentiles(times_ms: list[float]) -> dict:
    a = np.array(times_ms)
    return {
        "mean": a.mean(), "p50": np.percentile(a, 50),
        "p95": np.percentile(a, 95), "p99": np.percentile(a, 99),
        "max": a.max(),
    }


def report(name: str, times_ms: list[float], budget: float | None = None) -> dict:
    p = percentiles(times_ms)
    print(f"\n{name}")
    print(f"  mean {p['mean']:>8.2f} ms")
    print(f"  p50  {p['p50']:>8.2f} ms")
    print(f"  p95  {p['p95']:>8.2f} ms")
    print(f"  p99  {p['p99']:>8.2f} ms   <- the number that matters")
    print(f"  max  {p['max']:>8.2f} ms")
    if budget:
        verdict = "FITS" if p["p99"] < budget else "OVER BUDGET"
        print(f"  budget {budget} ms -> {verdict} "
              f"({p['p99'] / budget:.0%} of it used)")
    return {"stage": name, **p}


def time_calls(fn, n: int = N_SAMPLES, warmup: int = WARMUP) -> list[float]:
    for i in range(warmup):
        fn(i)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main() -> None:
    import lightgbm as lgb
    import xgboost as xgb

    X_test = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")
    print(f"card rail: {X_test.shape[1]} features\n")

    rows = []

    # --- model inference, batch 1 ------------------------------------------
    lgb_model = lgb.Booster(model_file=str(cfg.ARTIFACTS / "card_lightgbm.txt"))
    single = [X_test.iloc[[i % len(X_test)]] for i in range(N_SAMPLES + WARMUP)]

    rows.append(report("LightGBM inference (batch 1)",
                       time_calls(lambda i: lgb_model.predict(single[i]))))

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        tree_method="hist", n_jobs=1, random_state=cfg.RANDOM_SEED)
    y_small = np.load(cfg.INTERIM / "card_y_train.npy")[:20000]
    X_small = pd.read_parquet(cfg.INTERIM / "card_X_train.parquet").iloc[:20000]
    xgb_model.fit(X_small, y_small)
    rows.append(report("XGBoost inference (batch 1)",
                       time_calls(lambda i: xgb_model.predict_proba(single[i]))))

    # The shipped card model is both, rank-averaged.
    def blend(i):
        lgb_model.predict(single[i])
        xgb_model.predict_proba(single[i])
    rows.append(report("Blend inference (both models)",
                       time_calls(blend), budget=CARD_BUDGET_MS))

    # --- feature building, the part that usually dominates ------------------
    from features.card import CardFeatureBuilder
    from data import loaders
    from validation.splits import chronological_split

    df = loaders.load_card()
    split = chronological_split(df, test_fraction=0.20)
    train_df = df.iloc[split.train_idx]
    test_rows = df.iloc[split.test_idx]
    builder = CardFeatureBuilder().fit(train_df)
    del df, train_df

    singles_raw = [test_rows.iloc[[i % len(test_rows)]] for i in range(N_SAMPLES + WARMUP)]
    rows.append(report("Feature building (one transaction)",
                       time_calls(lambda i: builder.transform(singles_raw[i]))))

    # --- the whole path -----------------------------------------------------
    def end_to_end(i):
        features = builder.transform(singles_raw[i])
        lgb_model.predict(features)
        xgb_model.predict_proba(features)
    rows.append(report("END TO END (features + both models)",
                       time_calls(end_to_end), budget=CARD_BUDGET_MS))

    out = pd.DataFrame(rows)
    out.to_csv(cfg.REPORTS / "step6_latency.csv", index=False)
    print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
    print(out[["stage", "p50", "p99", "max"]].to_string(index=False))

    e2e = out.iloc[-1]
    print(f"\nEnd-to-end p99 is {e2e.p99:.1f} ms against a {CARD_BUDGET_MS} ms budget.")
    if e2e.p99 >= CARD_BUDGET_MS:
        print("OVER BUDGET. The fix is almost certainly feature building rather")
        print("than the models - see the per-stage numbers above.")


if __name__ == "__main__":
    main()
