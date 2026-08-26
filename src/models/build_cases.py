"""Turn wire-rail scores into investigable cases, and check whether they are any good.

`policy/cases.py` was written but never run end to end. This does that, and then
asks the question that matters: **do the traced cases actually correspond to the
370 labelled laundering rings?**

A case is only useful if an investigator opening it finds a real operation. So
three things get measured:

  1. **Case-level precision** - what share of cases contain any real laundering?
     This is the number an investigator experiences, and it is not the same as
     transaction-level precision: one case holding twelve alerts is one unit of
     work, not twelve.
  2. **Ring recall** - how many of the 370 labelled rings does the queue reach?
  3. **Evidence quality** - when a case is real, does the trace show a structure,
     or does it just say "no structure found"?

Run: python src/models/build_cases.py
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
from data.rings import load_rings, rings_to_frame
from features.graph_patterns import build_temporal_graph
from features.wire import LABEL, sort_by_time
from policy.cases import build_cases, case_detail, format_queue

ALERT_RATE = 0.01          # the wire rail's operating point
TOP_CASES_SHOWN = 15


def main() -> None:
    scores = np.load(cfg.INTERIM / "wire_seed_xgboost_42.npy")

    df = sort_by_time(loaders.load_wire())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")
    split = int(len(df) * 0.80)
    test_df = df.iloc[split:].copy()
    del df
    gc.collect()

    assert len(scores) == len(test_df), "scores and test rows are out of step"
    test_df["score"] = scores

    # --- the alert queue ---------------------------------------------------
    k = int(len(test_df) * ALERT_RATE)
    alerts = test_df.nlargest(k, "score")
    print(f"test set {len(test_df):,} wires, {int(test_df[LABEL].sum()):,} laundering")
    print(f"alerts at {ALERT_RATE:.0%}: {len(alerts):,} transactions, "
          f"{int(alerts[LABEL].sum()):,} of them laundering "
          f"({alerts[LABEL].mean():.1%} precision)\n", flush=True)

    # --- the graph the tracing runs on -------------------------------------
    t0 = time.time()
    graph = build_temporal_graph(test_df)
    print(f"graph built in {time.time() - t0:.0f}s: "
          f"{graph.number_of_nodes():,} accounts", flush=True)

    t0 = time.time()
    cases = build_cases(alerts, graph)
    print(f"{len(cases)} cases traced in {time.time() - t0:.0f}s\n", flush=True)

    # --- does a case contain real laundering? ------------------------------
    for case in cases:
        case.has_laundering = bool(case.transactions[LABEL].sum())

    real = sum(c.has_laundering for c in cases)
    print(f"{'=' * 66}\nCASE QUALITY\n{'=' * 66}")
    print(f"  cases                      {len(cases):>6,}")
    print(f"  containing real laundering {real:>6,}   "
          f"({real / len(cases):.1%} case precision)")
    print(f"  transaction precision      {alerts[LABEL].mean():>6.1%}   "
          f"(for comparison)")
    print(f"  alerts per case            {len(alerts) / len(cases):>6.1f}")

    # --- is the trace informative? -----------------------------------------
    with_structure = sum(1 for c in cases if c.cycles or c.chains)
    real_with_structure = sum(1 for c in cases
                              if c.has_laundering and (c.cycles or c.chains))
    print(f"\n  cases with a traced structure   {with_structure:>6,} "
          f"({with_structure / len(cases):.1%})")
    if real:
        print(f"  real cases with a structure     {real_with_structure:>6,} "
              f"({real_with_structure / real:.1%})")

    # --- how many labelled rings does the queue reach? ---------------------
    rings = load_rings()
    ring_df = rings_to_frame(rings)
    alert_accounts = set(alerts["Account"]) | set(alerts["Account.1"])
    reached = sum(1 for r in rings if r.accounts & alert_accounts)
    print(f"\n  labelled rings reached          {reached:>6,} of {len(rings)} "
          f"({reached / len(rings):.1%})")
    print("  (rings span the whole period; only the test fifth is scored here)")

    # --- does ranking by case beat ranking by transaction? ------------------
    print(f"\n{'=' * 66}\nTOP OF THE QUEUE\n{'=' * 66}")
    print(format_queue(cases, limit=TOP_CASES_SHOWN))

    top20 = cases[:20]
    hits = sum(c.has_laundering for c in top20)
    print(f"\n  of the top 20 cases, {hits} contain real laundering "
          f"({hits / len(top20):.0%})")

    # --- one worked example ------------------------------------------------
    example = next((c for c in cases if c.has_laundering and (c.cycles or c.chains)), None)
    if example is None:
        example = next((c for c in cases if c.has_laundering), None)
    if example is not None:
        print(f"\n{'=' * 66}\nWORKED EXAMPLE - what an investigator opens\n{'=' * 66}")
        print(case_detail(example))

    summary = pd.DataFrame([{
        "cases": len(cases), "alerts": len(alerts),
        "case_precision": real / len(cases),
        "transaction_precision": float(alerts[LABEL].mean()),
        "cases_with_structure": with_structure / len(cases),
        "rings_reached": reached, "rings_total": len(rings),
    }])
    summary.to_csv(cfg.REPORTS / "step5_case_quality.csv", index=False)
    print(f"\nwritten -> {cfg.REPORTS / 'step5_case_quality.csv'}")


if __name__ == "__main__":
    main()
