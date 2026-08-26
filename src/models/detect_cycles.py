"""Find laundering rings by looking for cycles in the account graph.

No model, no training, no labels. Build a directed graph of accounts, look for
money that returns to where it started within a time window, and see how much of
the labelled ground truth that alone recovers.

This runs before any GNN work on purpose. The question a GNN would answer is
"can a learned model find these shapes?" - but nobody has yet measured whether a
plain graph algorithm finds them. If `networkx.simple_cycles` already recovers
most of the 54 labelled CYCLE rings, a GNN is solving a problem that is largely
solved; if it recovers few, the GNN has a clear job.

Two things make this harder than textbook cycle detection:

  1. **Scale.** 5,078,345 transactions over 496,995 accounts. Enumerating all
     cycles in a graph that size does not terminate, so the search is bounded by
     length and restricted to a subgraph.
  2. **Time.** A -> B -> C -> A over two days is layering. The same three edges
     over two years is ordinary business. A cycle is only evidence if it closes
     quickly, so every candidate is checked against a time window.

Run: python src/models/detect_cycles.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data import loaders
from data.rings import load_rings, rings_to_frame

MAX_CYCLE_LENGTH = 6          # the labelled CYCLE rings have a median of 4 hops
TIME_WINDOW_DAYS = 30         # the whole dataset spans 18 days, so this is loose
MIN_ACCOUNT_DEGREE = 2        # an account with one edge cannot sit on a cycle


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """One node per account, one edge per account pair.

    Parallel transactions between the same pair collapse into a single edge
    carrying their count, total value and first/last timestamps - a multigraph
    would be more faithful but makes cycle enumeration far more expensive, and
    the timing test only needs the span.
    """
    grouped = df.groupby(["Account", "Account.1"]).agg(
        count=("Amount Paid", "size"),
        total=("Amount Paid", "sum"),
        first=("_ts", "min"),
        last=("_ts", "max"),
    ).reset_index()

    g = nx.DiGraph()
    for row in grouped.itertuples(index=False):
        g.add_edge(row[0], row[1], count=row.count, total=row.total,
                   first=row.first, last=row.last)
    return g


def prune(g: nx.DiGraph) -> nx.DiGraph:
    """Drop accounts that cannot participate in a cycle.

    A node needs at least one edge in and one out to sit on a cycle. Removing
    the rest repeatedly shrinks the search space enormously and cannot discard a
    real cycle.
    """
    g = g.copy()
    while True:
        dead = [n for n in g
                if g.in_degree(n) == 0 or g.out_degree(n) == 0]
        if not dead:
            return g
        g.remove_nodes_from(dead)


def find_cycles(g: nx.DiGraph, max_length: int) -> list[list]:
    """Enumerate simple cycles up to `max_length` hops."""
    return list(nx.simple_cycles(g, length_bound=max_length))


def cycle_closes_quickly(g: nx.DiGraph, cycle: list, days: int) -> bool:
    """Does the money go round within the window?

    Uses each edge's first and last timestamp: the cycle is judged by the span
    between the earliest and latest activity on it.
    """
    times = []
    for i, node in enumerate(cycle):
        nxt = cycle[(i + 1) % len(cycle)]
        edge = g.edges[node, nxt]
        times.extend([edge["first"], edge["last"]])
    span = (max(times) - min(times)).total_seconds() / 86400
    return span <= days


def main() -> None:
    print("loading transactions...", flush=True)
    df = loaders.load_wire()
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")

    rings = load_rings()
    ring_df = rings_to_frame(rings)
    cycle_rings = [r for r in rings if r.pattern == "CYCLE"]
    ring_accounts = set(ring_df.from_account) | set(ring_df.to_account)

    print(f"  {len(df):,} transactions, {df['Account'].nunique():,} accounts")
    print(f"  {len(rings)} labelled rings, {len(cycle_rings)} of them CYCLE")
    print(f"  {len(ring_accounts):,} accounts appear in some ring", flush=True)

    t0 = time.time()
    g = build_graph(df)
    print(f"\ngraph built in {time.time() - t0:.0f}s: "
          f"{g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges", flush=True)

    t0 = time.time()
    pruned = prune(g)
    print(f"pruned in {time.time() - t0:.0f}s to "
          f"{pruned.number_of_nodes():,} nodes, {pruned.number_of_edges():,} edges "
          f"({pruned.number_of_nodes() / g.number_of_nodes():.1%} of accounts kept)",
          flush=True)

    t0 = time.time()
    cycles = find_cycles(pruned, MAX_CYCLE_LENGTH)
    print(f"found {len(cycles):,} simple cycles up to {MAX_CYCLE_LENGTH} hops "
          f"in {time.time() - t0:.0f}s", flush=True)

    timed = [c for c in cycles if cycle_closes_quickly(pruned, c, TIME_WINDOW_DAYS)]
    print(f"  {len(timed):,} close within {TIME_WINDOW_DAYS} days", flush=True)

    # --- how much ground truth does this recover? -------------------------
    flagged_accounts = {n for c in timed for n in c}
    print(f"  {len(flagged_accounts):,} distinct accounts sit on a cycle")

    rows = []
    for ring in rings:
        hit = bool(ring.accounts & flagged_accounts)
        rows.append({"pattern": ring.pattern, "caught": hit})
    per_pattern = pd.DataFrame(rows).groupby("pattern").agg(
        rings=("caught", "size"), caught=("caught", "sum"))
    per_pattern["recall"] = per_pattern.caught / per_pattern.rings

    print(f"\n{'=' * 62}\nRING RECALL - does any account of the ring sit on a cycle?\n{'=' * 62}")
    print(per_pattern.sort_values("recall", ascending=False).to_string())
    print(f"\n  overall: {per_pattern.caught.sum()}/{per_pattern.rings.sum()} rings "
          f"({per_pattern.caught.sum() / per_pattern.rings.sum():.1%})")

    # --- and what does it cost? -------------------------------------------
    all_accounts = set(df["Account"]) | set(df["Account.1"])
    legit_accounts = all_accounts - ring_accounts
    false_positives = flagged_accounts & legit_accounts
    print(f"\n{'=' * 62}\nCOST\n{'=' * 62}")
    print(f"  accounts flagged        {len(flagged_accounts):>8,}")
    print(f"  of which in a real ring {len(flagged_accounts & ring_accounts):>8,}")
    print(f"  false positives         {len(false_positives):>8,}")
    precision = len(flagged_accounts & ring_accounts) / max(len(flagged_accounts), 1)
    print(f"  precision               {precision:>8.2%}")
    print(f"  share of all accounts   {len(flagged_accounts) / len(all_accounts):>8.2%}")

    per_pattern.to_csv(cfg.REPORTS / "step4_cycle_detection.csv")
    print(f"\nwritten -> {cfg.REPORTS / 'step4_cycle_detection.csv'}")


if __name__ == "__main__":
    main()
