"""Does the LLM panel improve the investigator queue?

The wire rail queues cases at ~12% precision. This measures whether a
majority-vote panel can raise that before a human sees it - and, just as
importantly, whether it throws away real laundering while doing so.

The second number is the one that decides adoption. A filter that doubles
precision by discarding a third of the real cases has made the system worse.

Run: python src/models/run_triage.py
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
from features.graph_patterns import build_temporal_graph
from features.wire import LABEL, sort_by_time
from policy.cases import build_cases
from policy.triage import format_triage, triage_queue

TOP_N = 40          # what an investigator would realistically work through


def main() -> None:
    scores = np.load(cfg.INTERIM / "wire_seed_xgboost_42.npy")
    df = sort_by_time(loaders.load_wire())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")
    test_df = df.iloc[int(len(df) * 0.80):].copy()
    test_df["score"] = scores
    del df
    gc.collect()

    alerts = test_df.nlargest(int(len(test_df) * 0.01), "score")
    graph = build_temporal_graph(test_df)
    cases = build_cases(alerts, graph)
    for case in cases:
        case.has_laundering = bool(case.transactions[LABEL].sum())

    print(f"{len(cases):,} cases; triaging the top {TOP_N}\n", flush=True)

    t0 = time.time()
    results = triage_queue(cases, limit=TOP_N)
    print(f"panel ran in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / TOP_N:.1f}s per case)\n")
    print(format_triage(results))

    truth = {c.case_id: c.has_laundering for c in cases[:TOP_N]}
    real = sum(truth.values())

    print(f"\n{'=' * 74}\nDID IT HELP?\n{'=' * 74}")
    print(f"  cases triaged        {len(results)}")
    print(f"  actually laundering  {real}  ({real / len(results):.0%} baseline precision)")

    for verdict in ("laundering", "unclear", "legitimate"):
        group = [r for r in results if r.verdict == verdict]
        if group:
            hits = sum(truth[r.case_id] for r in group)
            print(f"  panel says {verdict:<11} {len(group):>3} cases, "
                  f"{hits:>2} real ({hits / len(group):.0%})")

    escalated = [r for r in results if r.escalate]
    if escalated:
        hits = sum(truth[r.case_id] for r in escalated)
        missed = real - hits
        print(f"\n  escalated to a human  {len(escalated):>3} of {len(results)}")
        print(f"  precision after       {hits / len(escalated):>6.0%}  "
              f"(from {real / len(results):.0%})")
        print(f"  real cases NOT escalated {missed:>2}   <- the number that decides this")
        if missed:
            print("  A filter that raises precision by discarding real laundering")
            print("  has made the system worse, not better.")

    errors = sum(1 for r in results for rev in r.reviews if rev.error)
    if errors:
        print(f"\n  panel call failures: {errors} of {len(results) * 3}")

    pd.DataFrame([{**r.to_dict(), "actually_laundering": truth[r.case_id]}
                  for r in results]).to_csv(
        cfg.REPORTS / "step13_triage.csv", index=False)


if __name__ == "__main__":
    main()
