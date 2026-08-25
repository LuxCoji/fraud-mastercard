"""Choose how to split the 2% alert budget between blocking and stepping up.

The total alert rate is fixed at 2% (chosen in step 2). What is still open is
how much of that budget is spent on hard declines. This scores several splits
against the held-out test set so the choice is made on measured numbers.

Run: python src/policy/tune_bands.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from policy.bands import BLOCK, STEP_UP, evaluate_policy, fit_policy, format_policy

TOTAL_ALERT_RATE = 0.02
CANDIDATE_BLOCK_RATES = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02]


def main() -> None:
    scores = np.load(cfg.INTERIM / "card_scores_lightgbm.npy")
    y_test = np.load(cfg.INTERIM / "card_y_test.npy")
    print(f"test set: {len(y_test):,} transactions, {y_test.sum():,} fraud "
          f"({y_test.mean():.3%})\n")

    rows = []
    for block_rate in CANDIDATE_BLOCK_RATES:
        policy = fit_policy(scores, block_rate, TOTAL_ALERT_RATE)
        table = evaluate_policy(y_test, scores, policy)
        block = table[table.band == BLOCK].iloc[0]
        step = table[table.band == STEP_UP].iloc[0]
        rows.append({
            "block_rate": block_rate,
            "block_precision": block.precision,
            "fraud_blocked": block.fraud_recall,
            "genuine_blocked_pct": block.genuine_pct,
            "fraud_stepped_up": step.fraud_recall,
            "genuine_stepped_up_pct": step.genuine_pct,
            "total_fraud_caught": block.fraud_recall + step.fraud_recall,
        })

    out = pd.DataFrame(rows)
    print("Splitting a fixed 2% alert budget. Total fraud caught is the same")
    print("in every row - what changes is how much of it is a hard decline.\n")
    header = (f"{'block %':>8} {'precision':>10} {'fraud blocked':>14} "
              f"{'GENUINE BLOCKED':>16} {'fraud via OTP':>14}")
    print(header)
    print("-" * len(header))
    for _, r in out.iterrows():
        print(f"{r.block_rate:>7.2%} {r.block_precision:>10.1%} "
              f"{r.fraud_blocked:>13.1%} {r.genuine_blocked_pct:>15.3%} "
              f"{r.fraud_stepped_up:>13.1%}")

    out.to_csv(cfg.REPORTS / "step2_policy_bands.csv", index=False)

    # Single-threshold comparison: everything in the 2% budget is a decline.
    single = fit_policy(scores, TOTAL_ALERT_RATE, TOTAL_ALERT_RATE)
    single_table = evaluate_policy(y_test, scores, single)
    single_blocked = single_table[single_table.band == BLOCK].iloc[0]

    print(f"\n{'=' * 62}")
    print("For comparison - a single threshold, everything declined:")
    print(f"  genuine customers blocked: {single_blocked.genuine_pct:.3%}")

    # Recommend the largest block band that still declines under 1 genuine
    # customer in 2,000, which keeps hard declines rare while blocking the
    # cases the model is most certain about.
    safe = out[out.genuine_blocked_pct <= 0.0005]
    if len(safe):
        pick = safe.iloc[-1]
        reduction = 1 - pick.genuine_blocked_pct / single_blocked.genuine_pct
        print(f"\nRecommended: block the top {pick.block_rate:.2%}, step up the rest.")
        print(f"  genuine customers blocked: {pick.genuine_blocked_pct:.3%} "
              f"({reduction:.0%} fewer than a single threshold)")
        print(f"  fraud stopped outright:    {pick.fraud_blocked:.1%}")
        print(f"  fraud sent to a challenge: {pick.fraud_stepped_up:.1%}")

        policy = fit_policy(scores, float(pick.block_rate), TOTAL_ALERT_RATE)
        print(f"\n{format_policy(evaluate_policy(y_test, scores, policy), policy)}")


if __name__ == "__main__":
    main()
