"""Does blending LightGBM and XGBoost beat LightGBM alone?

Three seeds each, because the V-column experiment already showed that a single
seed on this data can point the wrong way by a full percent.

The decision rule is fixed before the results are seen:

    adopt the blend only if its mean PR-AUC beats LightGBM alone by more than
    the combined seed spread, AND it wins on every seed.

CatBoost was dropped after the first run: it scored 0.5472 against LightGBM's
0.5841 on seed 42, and adding it to the blend made things worse (0.6017 ->
0.5865). It cost 20 minutes a seed to make the answer worse.

Models are averaged by rank rather than by raw probability. Their scores are
calibrated differently, so averaging probabilities lets whichever model happens
to be most confident dominate; averaging ranks asks only where each model puts
a transaction relative to the others, which is what the operating points read.

Resumable: each model's test scores are written to data/interim as soon as they
exist, and any run whose file is already present is skipped. An interrupted run
therefore costs one model, not the whole experiment - the first attempt at this
was killed partway through and lost everything because it only saved at the end.

Run: python src/models/experiment_ensemble.py
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
from models.metrics import headline, operating_points
from models.train_card import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS

SEEDS = [42, 7, 2024]


def to_rank(scores: np.ndarray) -> np.ndarray:
    """Positions in 0..1, so models with different calibration combine fairly."""
    return scores.argsort().argsort() / (len(scores) - 1)


def fit_lightgbm(X_tr, y_tr, X_va, y_va, X_test, seed):
    import lightgbm as lgb
    params = dict(LGB_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    model = lgb.train(
        params, lgb.Dataset(X_tr, y_tr),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(250)],
    )
    return model.predict(X_test, num_iteration=model.best_iteration)


def fit_xgboost(X_tr, y_tr, X_va, y_va, X_test, seed):
    """XGBoost at depth 8 rather than Deotte's published depth 12.

    Depth 12 allows up to 4,096 leaves a tree and needs roughly 3-4 GB across a
    3,000-tree run. This machine has ~1.5 GB free and killed the process three
    times at exactly that point. Depth 8 caps a tree at 256 leaves, which fits.

    `QuantileDMatrix` is used instead of the sklearn wrapper because it bins the
    features once up front rather than holding a second full copy of the data.

    The question being answered is "does a second, different model help?", not
    "does Deotte's exact XGBoost help?". Depth 8 is still a strong model and
    still learns differently from LightGBM, which is what a blend needs. If a
    gain only appears at depth 12, it is a very expensive gain.
    """
    import xgboost as xgb

    params = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "max_depth": 8, "learning_rate": 0.02,
        "subsample": 0.8, "colsample_bytree": 0.4,
        "tree_method": "hist", "nthread": -1, "seed": seed,
    }
    dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, missing=np.nan)
    dvalid = xgb.QuantileDMatrix(X_va, label=y_va, missing=np.nan, ref=dtrain)

    booster = xgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=500,
    )
    print(f"    xgboost stopped at round {booster.best_iteration}", flush=True)
    dtest = xgb.DMatrix(X_test, missing=np.nan)
    return booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))


BUILDERS = {"lightgbm": fit_lightgbm, "xgboost": fit_xgboost}


def scores_path(model: str, seed: int) -> Path:
    return cfg.INTERIM / f"card_scores_{model}_seed{seed}.npy"


def get_scores(model, build, seed, data) -> np.ndarray:
    """Train, or reuse an earlier run's saved scores."""
    path = scores_path(model, seed)
    if path.exists():
        print(f"--- seed {seed} | {model}: reusing saved scores ---", flush=True)
        return np.load(path)

    print(f"--- seed {seed} | {model} ---", flush=True)
    t0 = time.time()
    scores = build(*data, seed)
    np.save(path, scores)          # saved before anything else can fail
    print(f"    trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    return scores


def main() -> None:
    X_train = pd.read_parquet(cfg.INTERIM / "card_X_train.parquet")
    X_test = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")
    y_train = np.load(cfg.INTERIM / "card_y_train.npy")
    y_test = np.load(cfg.INTERIM / "card_y_test.npy")

    cut = int(len(X_train) * 0.9)
    data = (X_train.iloc[:cut], y_train[:cut],
            X_train.iloc[cut:], y_train[cut:], X_test)
    print(f"fit on {cut:,}  validate on {len(X_train) - cut:,}  "
          f"test on {len(X_test):,} ({y_test.sum():,} fraud)\n", flush=True)

    results_csv = cfg.REPORTS / "step2_ensemble_experiment.csv"
    rows: list[dict] = []

    def record(seed: int, model: str, scores: np.ndarray) -> None:
        h = headline(y_test, scores)
        points = operating_points(y_test, scores)
        at_two = points[points.alert_rate == 0.02].iloc[0]
        rows.append({"seed": seed, "model": model, **h,
                     "recall_at_2pct": at_two.recall,
                     "genuine_hit_at_2pct": at_two.genuine_disturbed_pct})
        # Rewritten after every model, so a kill costs one run at most.
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        print(f"    {model:16s} PR-AUC {h['pr_auc']:.4f}  AUC {h['auc']:.4f}  "
              f"recall@2% {at_two.recall:.2%}", flush=True)

    for seed in SEEDS:
        scores = {}
        for model, build in BUILDERS.items():
            scores[model] = get_scores(model, build, seed, data)
            record(seed, model, scores[model])
            gc.collect()

        blended = np.mean([to_rank(s) for s in scores.values()], axis=0)
        np.save(scores_path("blend_lgb_xgb", seed), blended)
        record(seed, "blend_lgb_xgb", blended)
        print(flush=True)

    out = pd.DataFrame(rows)
    summary = out.groupby("model").agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        auc_mean=("auc", "mean"), recall_mean=("recall_at_2pct", "mean"),
    ).sort_values("pr_auc_mean", ascending=False)
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(summary.to_string())

    print(f"\n{'=' * 70}\nPER SEED\n{'=' * 70}")
    print(out.pivot(index="seed", columns="model", values="pr_auc").to_string())

    base = out[out.model == "lightgbm"].sort_values("seed").pr_auc.values
    print(f"\n{'=' * 70}\nVERDICT (vs LightGBM alone)\n{'=' * 70}")
    for label in ("blend_lgb_xgb", "xgboost"):
        challenger = out[out.model == label].sort_values("seed").pr_auc.values
        gap = challenger.mean() - base.mean()
        spread = np.sqrt(base.std(ddof=1) ** 2 + challenger.std(ddof=1) ** 2)
        every = bool((challenger > base).all())
        real = gap > spread and every
        print(f"  {label:16s} gap {gap:+.4f} ({gap / base.mean():+.2%})  "
              f"spread {spread:.4f}  wins all seeds {str(every):5s}  "
              f"-> {'REAL' if real else 'NOT PROVEN'}")


if __name__ == "__main__":
    main()
