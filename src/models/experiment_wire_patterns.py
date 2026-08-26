"""Do explicit pattern counts beat aggregate account statistics?

The wire rail currently scores PR-AUC 0.1048 on account aggregates - degrees,
totals, pass-through ratios. The IBM paper behind this dataset attributes its
much stronger gradient-boosted result to counting the laundering *shapes*
directly, so this measures that claim on our own split.

Decision rule, fixed before results: adopt the pattern features only if they
improve PR-AUC by more than the seed spread measured on the card rail (about
0.4% relative) - and report the number either way.

Run: python src/models/experiment_wire_patterns.py
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


def run(with_patterns: bool, label: str, train_df, test_df, y_train, y_test,
        with_chains: bool = False) -> dict:
    import lightgbm as lgb

    t0 = time.time()
    builder = WireFeatureBuilder(with_patterns=with_patterns,
                                 with_chains=with_chains).fit(train_df)
    X_train = builder.transform(train_df)
    X_test = builder.transform(test_df)
    build_mins = (time.time() - t0) / 60
    print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
    print(f"  features {X_train.shape[1]}  (built in {build_mins:.1f} min)", flush=True)

    cut = int(len(X_train) * 0.9)
    model = lgb.train(
        dict(LGB_PARAMS, seed=cfg.RANDOM_SEED),
        lgb.Dataset(X_train.iloc[:cut], y_train[:cut]),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_train.iloc[cut:], y_train[cut:])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(200)],
    )
    scores = model.predict(X_test, num_iteration=model.best_iteration)

    h = headline(y_test, scores)
    points = operating_points(y_test, scores)
    at_one = points[points.alert_rate == 0.01].iloc[0]
    print(f"  rounds  {model.best_iteration}")
    print(f"  PR-AUC  {h['pr_auc']:.4f}")
    print(f"  AUC     {h['auc']:.4f}")
    print()
    print(format_operating_points(points))

    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
    }).sort_values("gain", ascending=False)

    if with_patterns:
        pattern_cols = [c for c in X_train.columns
                        if any(k in c for k in ("cycle", "scatter", "gather",
                                                "two_hop", "burst", "chain",
                                                "bipartite"))]
        share = importance[importance.feature.isin(pattern_cols)].gain.sum() / importance.gain.sum()
        print(f"\n  pattern-count features carry {share:.1%} of total gain")
        print("\n  top 12 features:")
        print(importance.head(12).to_string(index=False))
        np.save(cfg.INTERIM / "wire_scores_patterns.npy", scores)
        importance.to_csv(cfg.REPORTS / "step4_wire_pattern_importance.csv", index=False)

    n_features = X_train.shape[1]
    del X_train, X_test, model, builder
    gc.collect()
    return {"variant": label, "features": n_features, "pr_auc": h["pr_auc"],
            "auc": h["auc"], "brier": h["brier"],
            "recall_at_1pct": at_one.recall,
            "precision_at_1pct": at_one.precision}


def main() -> None:
    df = sort_by_time(loaders.load_wire())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")
    cut = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    y_train = train_df[LABEL].values
    y_test = test_df[LABEL].values
    print(f"train {len(train_df):,} rows ({int(y_train.sum()):,} laundering)  "
          f"test {len(test_df):,} rows ({int(y_test.sum()):,} laundering)", flush=True)
    del df
    gc.collect()

    rows = [
        run(True, "PATTERN COUNTS (current best)", train_df, test_df, y_train, y_test),
        run(True, "+ CHAIN & BIPARTITE", train_df, test_df, y_train, y_test,
            with_chains=True),
    ]

    out = pd.DataFrame(rows)
    out.to_csv(cfg.REPORTS / "step4_wire_pattern_experiment.csv", index=False)
    print(f"\n{'=' * 70}\nCOMPARISON\n{'=' * 70}")
    print(out.to_string(index=False))

    gain = (rows[1]["pr_auc"] - rows[0]["pr_auc"]) / rows[0]["pr_auc"]
    print(f"\nPR-AUC change from pattern counts: {gain:+.2%}")
    print("Seed noise on the card rail was about 0.4% relative; treat anything")
    print("smaller than that as unproven.")


if __name__ == "__main__":
    main()
