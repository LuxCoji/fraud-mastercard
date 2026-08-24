"""Train the card-rail models and report what they actually buy.

Two rungs, in order:

  1. Logistic regression - the floor. In the previous project a ten-line
     scikit-learn baseline beat a behavioural autoencoder on two of three
     sources, so nothing more complicated is worth its latency until it has
     cleared this bar.
  2. LightGBM with the reference notebooks' published hyperparameters.

Run: python src/models/train_card.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from models.metrics import format_operating_points, headline, operating_points

# Published in kaggle.com/kyakovlev/ieee-lgbm-with-groupkfold-cv, unchanged
# except for two things: no GPU here, and the tree count is left to early
# stopping because the split differs from his.
LGB_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "learning_rate": 0.01,
    "num_leaves": 2 ** 8,
    "max_depth": -1,
    "colsample_bytree": 0.5,
    "subsample": 0.7,
    "subsample_freq": 1,
    "max_bin": 255,
    "verbose": -1,
    "num_threads": 0,
    "seed": cfg.RANDOM_SEED,
}
MAX_ROUNDS = 3000
EARLY_STOPPING = 100


def load_features():
    X_train = pd.read_parquet(cfg.INTERIM / "card_X_train.parquet")
    X_test = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")
    y_train = np.load(cfg.INTERIM / "card_y_train.npy")
    y_test = np.load(cfg.INTERIM / "card_y_test.npy")
    return X_train, X_test, y_train, y_test


def train_logistic(X_train, y_train, X_test):
    """Floor model: median-imputed, scaled, unregularised-ish logistic.

    No class weighting. Rebalancing shifts every predicted probability toward
    0.5, which leaves ranking metrics almost unchanged while destroying the
    calibration that the policy thresholds depend on.
    """
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


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Reference-notebook LightGBM.

    The last 10% of the training period is held out as a validation tail for
    early stopping, so the test set is never consulted during fitting.
    """
    import lightgbm as lgb

    cut = int(len(X_train) * 0.9)
    fit_set = lgb.Dataset(X_train.iloc[:cut], y_train[:cut])
    val_set = lgb.Dataset(X_train.iloc[cut:], y_train[cut:])

    model = lgb.train(
        LGB_PARAMS, fit_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    print(f"  early stopping chose {model.best_iteration} rounds "
          f"(published setting was 800)")
    return model.predict(X_test, num_iteration=model.best_iteration), model


def report(name: str, y_test, scores) -> dict:
    h = headline(y_test, scores)
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  AUC     {h['auc']:.4f}")
    print(f"  PR-AUC  {h['pr_auc']:.4f}    (base rate {h['base_rate']:.3%})")
    print(f"  Brier   {h['brier']:.5f}  vs baseline {h['brier_baseline']:.5f}"
          f"   {'OK' if h['brier'] <= h['brier_baseline'] else 'WORSE THAN BASELINE'}")
    print()
    print(format_operating_points(operating_points(y_test, scores)))
    return h


def main() -> None:
    X_train, X_test, y_train, y_test = load_features()
    print(f"train {X_train.shape}   test {X_test.shape}   "
          f"test fraud {y_test.sum():,} ({y_test.mean():.3%})")

    results = {}

    t0 = time.time()
    scores_lr = train_logistic(X_train, y_train, X_test)
    print(f"\nlogistic trained in {time.time() - t0:.0f}s")
    results["logistic"] = report("RUNG 1 - Logistic regression (the floor)", y_test, scores_lr)

    t0 = time.time()
    scores_lgb, model = train_lightgbm(X_train, y_train, X_test, y_test)
    print(f"\nlightgbm trained in {time.time() - t0:.0f}s")
    results["lightgbm"] = report("RUNG 2 - LightGBM (notebook hyperparameters)", y_test, scores_lgb)

    np.save(cfg.INTERIM / "card_scores_logistic.npy", scores_lr)
    np.save(cfg.INTERIM / "card_scores_lightgbm.npy", scores_lgb)
    model.save_model(str(cfg.ARTIFACTS / "card_lightgbm.txt"))

    operating_points(y_test, scores_lgb).to_csv(
        cfg.REPORTS / "step2_card_operating_points.csv", index=False)

    lift = results["lightgbm"]["pr_auc"] / results["logistic"]["pr_auc"]
    print(f"\n{'=' * 62}")
    print(f"LightGBM PR-AUC is {lift:.2f}x the logistic floor.")
    print("If that number were near 1.0, the extra model would not be earning "
          "its latency.")

    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(cfg.REPORTS / "step2_card_feature_importance.csv", index=False)
    print(f"\nTop 15 features by gain:")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
