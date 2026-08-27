"""Does training on value rather than count catch the large frauds?

Measured on the shipped card model: it catches 45.6% of frauds by count but only
30.7% by value, and the detection rate falls monotonically with amount -

    smallest decile (2-16)      69.9% caught
    largest decile (335-3,133)  24.3% caught
    the 20 largest frauds        1 of 20 caught

That is backwards. The model was trained to maximise PR-AUC, which counts a
2-rupee fraud and a 3,000-rupee fraud as one event each, so it optimises for the
many small ones.

The fix under test is the obvious one: weight each fraud by its amount during
training, so a large fraud contributes proportionally more to the loss.

The risk is symmetric and must be measured, not assumed: making the model
keener on large amounts also makes its *mistakes* larger. `genuine_value_
disturbed` is reported alongside, because a model that stops more fraud value by
disturbing even more legitimate value has not improved anything.

Run: python src/models/experiment_value_weighted.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import config as cfg
from models.metrics import headline, operating_points, value_operating_points

ALERT_RATE = 0.02


def train(X_train, y_train, X_test, weights=None):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.4, missing=np.nan, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=cfg.RANDOM_SEED)
    model.fit(X_train, y_train, sample_weight=weights)
    return model.predict_proba(X_test)[:, 1]


def report(name: str, y_test, scores, amounts) -> dict:
    h = headline(y_test, scores)
    counts = operating_points(y_test, scores)
    values = value_operating_points(y_test, scores, amounts)
    c = counts[counts.alert_rate == ALERT_RATE].iloc[0]
    v = values[values.alert_rate == ALERT_RATE].iloc[0]

    fraud = y_test == 1
    thr = float(np.quantile(scores, 1 - ALERT_RATE))
    largest = np.argsort(-amounts[fraud])[:20]
    top20_caught = int((scores[fraud][largest] >= thr).sum())

    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  PR-AUC                    {h['pr_auc']:.4f}")
    print(f"  fraud caught by count     {c.recall:.1%}")
    print(f"  fraud caught by VALUE     {v.value_recall:.1%}")
    print(f"  fraud value stopped       {v.fraud_value_stopped:>12,.0f}")
    print(f"  genuine value disturbed   {v.genuine_value_disturbed:>12,.0f}")
    print(f"  value cost ratio          {v.value_cost_ratio:.2f}")
    print(f"  of the 20 largest frauds  {top20_caught} caught")
    return {"variant": name, "pr_auc": h["pr_auc"], "count_recall": c.recall,
            "value_recall": v.value_recall,
            "value_stopped": v.fraud_value_stopped,
            "genuine_disturbed": v.genuine_value_disturbed,
            "top20_caught": top20_caught}


def main() -> None:
    X_train = pd.read_parquet(cfg.INTERIM / "card_X_train.parquet")
    X_test = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")
    y_train = np.load(cfg.INTERIM / "card_y_train.npy")
    y_test = np.load(cfg.INTERIM / "card_y_test.npy")
    train_amounts = X_train["TransactionAmt"].to_numpy(float)
    test_amounts = X_test["TransactionAmt"].to_numpy(float)

    print(f"train {X_train.shape}  test {X_test.shape}")
    print(f"fraud value in test: {test_amounts[y_test == 1].sum():,.0f}")

    rows = [report("BASELINE - unweighted", y_test,
                   train(X_train, y_train, X_test), test_amounts)]

    # Weight by amount, normalised so the average weight stays 1. Without
    # normalising, the effective learning rate changes with the currency scale.
    weights = train_amounts / train_amounts.mean()
    rows.append(report("VALUE-WEIGHTED - weight = amount", y_test,
                       train(X_train, y_train, X_test, weights), test_amounts))

    # Square root compresses the range. Raw amounts are heavy-tailed, so a
    # handful of very large transactions would otherwise dominate the loss
    # entirely and the model would learn little else.
    sqrt_weights = np.sqrt(train_amounts) / np.sqrt(train_amounts).mean()
    rows.append(report("VALUE-WEIGHTED - weight = sqrt(amount)", y_test,
                       train(X_train, y_train, X_test, sqrt_weights), test_amounts))

    out = pd.DataFrame(rows)
    out.to_csv(cfg.REPORTS / "step10_value_weighted.csv", index=False)
    print(f"\n{'=' * 78}\nCOMPARISON\n{'=' * 78}")
    print(out.to_string(index=False))

    base = rows[0]
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    for row in rows[1:]:
        value_gain = (row["value_recall"] - base["value_recall"]) / base["value_recall"]
        cost_gain = (row["genuine_disturbed"] - base["genuine_disturbed"]) / base["genuine_disturbed"]
        print(f"  {row['variant']}")
        print(f"    fraud value stopped   {value_gain:+.1%}")
        print(f"    genuine value touched {cost_gain:+.1%}   <- the cost side")
        print(f"    worth it: {value_gain > cost_gain}")


if __name__ == "__main__":
    main()
