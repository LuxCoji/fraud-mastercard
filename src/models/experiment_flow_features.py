"""Does the flow tracer help the model, or only the investigator?

The tracer reaches **79% of the labelled rings** against 19% for the
account-graph detectors it replaced - but it was only ever wired into case
display. The wire model uses 49 features and not one of them comes from it.

That is the gap this measures. The same paths, summarised per account:

    flow_count          how many traced flows an account appears in
    flow_cycles         how many of those return to their origin
    flow_depth          the furthest position it occupies in any flow
    flow_as_middle      times it sits between two other accounts
    flow_middle_ratio   the share of its appearances that are in the middle
    flow_fastest_hours  the quickest flow it belongs to

Position is the point. An account that only starts flows is a source, one that
only ends them is a destination, and one that sits in the middle of long fast
flows is a conduit - which is what layering looks like from outside. A degree
count cannot express that.

Decision rule, fixed before results: adopt only if recall at the 1% operating
point improves by more than the seed spread measured on this rail (1.2%). PR-AUC
is reported but does not decide - it varied 36.7% across seeds here while recall
varied 1.2%.

Run: python src/models/experiment_flow_features.py
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
warnings.filterwarnings("ignore")

import config as cfg
from data import loaders
from features.wire import LABEL, WireFeatureBuilder, sort_by_time
from models.metrics import format_operating_points, headline, operating_points

SEEDS = [42, 7, 2024]
MAX_ROUNDS, EARLY_STOPPING = 3000, 100


def train_one(X_fit, y_fit, X_stop, y_stop, X_test, seed):
    import xgboost as xgb
    params = {"objective": "binary:logistic", "eval_metric": "aucpr",
              "max_depth": 8, "learning_rate": 0.02, "subsample": 0.8,
              "colsample_bytree": 0.4, "tree_method": "hist",
              "nthread": -1, "seed": seed}
    dtrain = xgb.QuantileDMatrix(X_fit, label=y_fit, missing=np.nan)
    dvalid = xgb.QuantileDMatrix(X_stop, label=y_stop, missing=np.nan, ref=dtrain)
    bst = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                    evals=[(dvalid, "valid")],
                    early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
    scores = bst.predict(xgb.DMatrix(X_test, missing=np.nan),
                         iteration_range=(0, bst.best_iteration + 1))
    return scores, bst


def run(with_flows: bool, label: str, train_df, test_df, y_train, y_test) -> list[dict]:
    fit_end = int(len(train_df) * 0.9)
    t0 = time.time()
    builder = WireFeatureBuilder(with_patterns=True,
                                 with_flows=with_flows).fit(train_df.iloc[:fit_end])
    X_fit = builder.transform(train_df.iloc[:fit_end])
    X_stop = builder.transform(train_df.iloc[fit_end:])
    X_test = builder.transform(test_df)
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    print(f"  {X_fit.shape[1]} features, built in {(time.time() - t0) / 60:.1f} min",
          flush=True)

    rows, last = [], None
    for seed in SEEDS:
        scores, bst = train_one(X_fit, y_train[:fit_end], X_stop,
                                y_train[fit_end:], X_test, seed)
        h = headline(y_test, scores)
        points = operating_points(y_test, scores)
        at_one = points[points.alert_rate == 0.01].iloc[0]
        rows.append({"variant": label, "seed": seed, "features": X_fit.shape[1],
                     "pr_auc": h["pr_auc"], "recall_at_1pct": at_one.recall,
                     "precision_at_1pct": at_one.precision,
                     "rounds": bst.best_iteration})
        # Written every seed, so a kill costs one training rather than all six.
        path = cfg.REPORTS / "step14_flow_features.csv"
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([existing[existing.variant != label] if len(existing) else existing,
                   pd.DataFrame(rows)]).to_csv(path, index=False)
        degenerate = bst.best_iteration < 10
        print(f"  seed {seed:>5}: recall@1% {at_one.recall:>6.2%}  "
              f"PR-AUC {h['pr_auc']:.4f}  ({bst.best_iteration} rounds)"
              f"{'   <- DEGENERATE, excluded' if degenerate else ''}", flush=True)
        rows[-1]["degenerate"] = degenerate
        last = (scores, bst, X_fit.columns)

    if with_flows and last:
        scores, bst, columns = last
        gains = bst.get_score(importance_type="gain")
        total = sum(gains.values())
        flow_gain = sum(v for k, v in gains.items() if "flow_" in k)
        print(f"\n  flow features carry {flow_gain / total:.1%} of total gain")
        top = sorted(gains.items(), key=lambda kv: -kv[1])[:10]
        print("  top 10 features:")
        for name, gain in top:
            mark = "  <- flow" if "flow_" in name else ""
            print(f"    {name:<28} {gain:>12,.0f}{mark}")
        print()
        print(format_operating_points(operating_points(y_test, scores)))

    del X_fit, X_stop, X_test, builder
    gc.collect()
    return rows


def main() -> None:
    df = sort_by_time(loaders.load_wire())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")
    split = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    y_train, y_test = train_df[LABEL].values, test_df[LABEL].values
    print(f"train {len(train_df):,}  test {len(test_df):,} "
          f"({int(y_test.sum()):,} laundering)", flush=True)
    del df
    gc.collect()

    rows = run(False, "PATTERNS ONLY (shipped)", train_df, test_df, y_train, y_test)
    rows += run(True, "PATTERNS + TRACED FLOWS", train_df, test_df, y_train, y_test)

    out = pd.DataFrame(rows)
    out.to_csv(cfg.REPORTS / "step14_flow_features.csv", index=False)

    summary = out.groupby("variant").agg(
        features=("features", "first"),
        recall=("recall_at_1pct", "mean"), recall_sd=("recall_at_1pct", "std"),
        pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"))
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(summary.to_string())

    # A seed that early-stopped in under ten rounds produced a stump, not a
    # model. One did in the previous run - it stopped at round 1 and scored
    # 47.02% - and letting it set the baseline would make any change look good.
    usable = out[~out.get("degenerate", False).fillna(False)]
    dropped = len(out) - len(usable)
    if dropped:
        print(f"\n  excluded {dropped} degenerate run(s) from the comparison")
    base = usable[usable.variant.str.startswith("PATTERNS ONLY")].sort_values("seed")
    test = usable[usable.variant.str.startswith("PATTERNS +")].sort_values("seed")
    if len(base) != len(test):
        print(f"  WARNING: {len(base)} baseline vs {len(test)} test runs - "
              "the seeds no longer pair, so this is not a like-for-like comparison")
    gap = test.recall_at_1pct.mean() - base.recall_at_1pct.mean()
    spread = np.sqrt(base.recall_at_1pct.std(ddof=1) ** 2 +
                     test.recall_at_1pct.std(ddof=1) ** 2)
    every = bool(len(base) == len(test) and
                 (test.recall_at_1pct.values > base.recall_at_1pct.values).all())

    print(f"\n{'=' * 70}\nVERDICT (recall at the 1% operating point)\n{'=' * 70}")
    print(f"  mean gap        {gap:+.4f}  ({gap / base.recall_at_1pct.mean():+.1%})")
    print(f"  combined spread {spread:.4f}")
    print(f"  wins every seed {every}")
    print(f"\n  -> {'ADOPT' if gap > spread and every else 'NOT PROVEN'}")


if __name__ == "__main__":
    main()
