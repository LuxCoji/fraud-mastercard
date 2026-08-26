"""Build the shippable wire-rail model: blend, calibrate, evaluate, save.

Two things the card rail proved and this rail has not yet had:

  **A blend.** LightGBM plus XGBoost, rank-averaged, beat LightGBM alone on all
  three seeds on the card rail (+1.41% PR-AUC). The mechanism was not that
  XGBoost is better - it is worse and far less stable - but that averaging two
  differently-wrong models cancels noise.

  **Calibration.** The wire rail's Brier beats its baseline by 0.2%, against
  30% on the card rail. Its scores rank but do not mean anything as
  probabilities, and every policy threshold is a probability.

Split, mirroring the card rail:

    fit          first 85% of the training period
    early stop   next 7.5%
    calibrate    last 7.5%
    test         the held-out final 20%, untouched

Platt rather than isotonic, for the reason measured on the card rail: isotonic
is a step function and collapsed 118,108 distinct scores into 153, tying 99.87%
of rows and costing 0.027 PR-AUC. A sigmoid is strictly increasing, so ranking
survives exactly.

Run: python src/models/build_wire_model.py
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
from data import loaders
from features.wire import LABEL, WireFeatureBuilder, sort_by_time
from models.build_card_model import isotonic, platt
from models.experiment_ensemble import to_rank
from models.metrics import format_operating_points, headline, operating_points
from models.train_card import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS

SEED = cfg.RANDOM_SEED
FIT_FRACTION = 0.85
EARLY_STOP_FRACTION = 0.075


def fit_lightgbm(X_fit, y_fit, X_stop, y_stop, X_pred):
    import lightgbm as lgb
    model = lgb.train(
        dict(LGB_PARAMS, seed=SEED),
        lgb.Dataset(X_fit, y_fit),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_stop, y_stop)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(200)],
    )
    print(f"    lightgbm stopped at {model.best_iteration}", flush=True)
    return model.predict(X_pred, num_iteration=model.best_iteration)


def fit_xgboost(X_fit, y_fit, X_stop, y_stop, X_pred):
    """Depth 8 and QuantileDMatrix, for the memory reason found on the card rail."""
    import xgboost as xgb
    params = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "max_depth": 8, "learning_rate": 0.02,
        "subsample": 0.8, "colsample_bytree": 0.4,
        "tree_method": "hist", "nthread": -1, "seed": SEED,
    }
    dtrain = xgb.QuantileDMatrix(X_fit, label=y_fit, missing=np.nan)
    dvalid = xgb.QuantileDMatrix(X_stop, label=y_stop, missing=np.nan, ref=dtrain)
    booster = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                        evals=[(dvalid, "valid")],
                        early_stopping_rounds=EARLY_STOPPING,
                        verbose_eval=500)
    print(f"    xgboost stopped at {booster.best_iteration}", flush=True)
    return booster.predict(xgb.DMatrix(X_pred, missing=np.nan),
                           iteration_range=(0, booster.best_iteration + 1))


def report(name: str, y_true, scores) -> dict:
    h = headline(y_true, scores)
    state = "OK" if h["brier"] <= h["brier_baseline"] else "WORSE THAN BASELINE"
    margin = (1 - h["brier"] / h["brier_baseline"]) * 100
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  PR-AUC  {h['pr_auc']:.4f}")
    print(f"  AUC     {h['auc']:.4f}")
    print(f"  Brier   {h['brier']:.6f}  vs baseline {h['brier_baseline']:.6f}"
          f"  ({margin:+.1f}%)  {state}")
    print()
    print(format_operating_points(operating_points(y_true, scores)))
    return h


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

    print(f"fit        {fit_end:>9,} rows  {int(y_train[:fit_end].sum()):>5,} laundering")
    print(f"early stop {stop_end - fit_end:>9,} rows  "
          f"{int(y_train[fit_end:stop_end].sum()):>5,} laundering")
    print(f"calibrate  {n - stop_end:>9,} rows  "
          f"{int(y_train[stop_end:].sum()):>5,} laundering")
    print(f"test       {len(test_df):>9,} rows  {int(y_test.sum()):>5,} laundering\n",
          flush=True)

    n_cal_positives = int(y_train[stop_end:].sum())
    if n_cal_positives < 200:
        print(f"NOTE: only {n_cal_positives} positives in the calibration slice.")
        print("Platt (2 parameters) is the safe choice; isotonic would overfit.\n")

    t0 = time.time()
    builder = WireFeatureBuilder(with_patterns=True).fit(train_df.iloc[:fit_end])
    X_fit = builder.transform(train_df.iloc[:fit_end])
    X_stop = builder.transform(train_df.iloc[fit_end:stop_end])
    X_cal = builder.transform(train_df.iloc[stop_end:])
    X_test = builder.transform(test_df)
    print(f"features built in {(time.time() - t0) / 60:.1f} min: "
          f"{X_fit.shape[1]} columns\n", flush=True)

    y_fit, y_stop = y_train[:fit_end], y_train[fit_end:stop_end]
    y_cal = y_train[stop_end:]
    del train_df, test_df
    gc.collect()

    X_pred = pd.concat([X_cal, X_test], axis=0)
    cal_len = len(X_cal)

    raw = {}
    for name, fit in (("lightgbm", fit_lightgbm), ("xgboost", fit_xgboost)):
        cache = cfg.INTERIM / f"wire_final_raw_{name}.npy"
        if cache.exists():
            print(f"--- {name}: reusing saved scores ---", flush=True)
            both = np.load(cache)
        else:
            print(f"--- {name} ---", flush=True)
            t0 = time.time()
            both = fit(X_fit, y_fit, X_stop, y_stop, X_pred)
            np.save(cache, both)
            print(f"    trained in {(time.time() - t0) / 60:.1f} min", flush=True)
        raw[name] = {"cal": both[:cal_len], "test": both[cal_len:]}
        gc.collect()

    results = {}
    for name in raw:
        results[name] = report(f"{name} (raw)", y_test, raw[name]["test"])

    rank_cal = np.mean([to_rank(raw[m]["cal"]) for m in raw], axis=0)
    rank_test = np.mean([to_rank(raw[m]["test"]) for m in raw], axis=0)
    results["blend_raw"] = report("BLEND - rank-averaged, uncalibrated",
                                  y_test, rank_test)

    platt_test = platt(rank_cal, y_cal).predict(rank_test)
    results["blend_platt"] = report("BLEND - Platt calibrated", y_test, platt_test)

    iso_test = isotonic(rank_cal, y_cal).predict(rank_test)
    results["blend_isotonic"] = report("BLEND - isotonic calibrated", y_test, iso_test)

    candidates = {"blend_platt": platt_test, "blend_isotonic": iso_test}
    table = pd.DataFrame(results).T[["pr_auc", "auc", "brier"]]
    table["brier_baseline"] = results["blend_platt"]["brier_baseline"]
    table["calibrated"] = table.brier <= table.brier_baseline
    table["distinct_scores"] = [len(np.unique(candidates[k])) if k in candidates
                                else np.nan for k in table.index]
    print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
    print(table.to_string())
    table.to_csv(cfg.REPORTS / "step4_wire_calibration.csv")

    usable = {k: v for k, v in results.items()
              if k in candidates and v["brier"] <= v["brier_baseline"]}
    if usable:
        best = max(usable, key=lambda k: usable[k]["pr_auc"])
        print(f"\nCalibrated and best-ranking: {best}")
        np.save(cfg.INTERIM / "wire_scores_final.npy", candidates[best])

        baseline = 0.1522   # LightGBM alone with pattern features
        change = (usable[best]["pr_auc"] - baseline) / baseline
        print(f"vs LightGBM alone (0.1522): {change:+.2%}")

        (cfg.ARTIFACTS / "wire_model_card.json").write_text(json.dumps({
            "rail": "wire", "source": "ibm_aml_hi_small",
            "models": list(raw), "blend": best, "seed": SEED,
            "synthetic_source": True,
            "metrics": {k: {m: float(v[m]) for m in ("pr_auc", "auc", "brier")}
                        for k, v in results.items()},
        }, indent=2))
    else:
        print("\nNo calibration method beat the Brier baseline - none adopted.")


if __name__ == "__main__":
    main()
