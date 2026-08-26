"""Pattern-counting graph features for the wire rail.

The first attempt at graph features counted only what an account does in
aggregate - how many partners, how much passes through. That reached PR-AUC
0.1048. The IBM paper that produced this dataset reports a much stronger result
from the same model family (LightGBM at 62.9% minority-class F1, beating every
GNN they tested), and the difference is not the model: it is that their Graph
Feature Preprocessor counts *the patterns themselves*.

So this module answers, per account: how many tight cycles do you sit on? How
many scatter-gather structures? How concentrated is your fan-out?

Two design points taken directly from the paper, both of which the first
cycle-detection attempt got wrong:

  **Time windows must be tight.** The paper uses six hours for scatter-gather
  and one day for cycles. The first attempt used thirty days on an
  eighteen-day dataset, so every cycle trivially passed and the filter did
  nothing - 372,952 cycles survived and 71.6% of all accounts were flagged.

  **Count, do not flag.** "This account is on a cycle" is nearly useless:
  ordinary business forms loops constantly. "This account is on three cycles
  that each close within a day" is a different statement. The features here are
  all counts and ratios, never booleans, so the model can learn where the
  threshold lies rather than having one imposed.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

# From the paper: six hours for scatter-gather, one day for cycles.
CYCLE_WINDOW_HOURS = 24
SCATTER_GATHER_WINDOW_HOURS = 6
MAX_CYCLE_LENGTH = 6


def build_temporal_graph(df: pd.DataFrame, min_account_txns: int = 1) -> nx.DiGraph:
    """Account graph with timing on every edge.

    Parallel transactions between one pair collapse to a single edge that keeps
    their count, value and the span between first and last - which is what the
    window tests need.

    `min_account_txns` drops accounts appearing fewer than that many times
    before the graph is built. On HI-Medium - 31.9M transactions over roughly
    three million accounts - building the full graph and enumerating its cycles
    exhausted 12 GB and was killed. The long tail of accounts seen once or
    twice cannot sit on a ring of any interesting size, so excluding them costs
    almost nothing and is what makes the larger datasets tractable at all.

    Left at 1 by default so existing results stay reproducible.
    """
    if min_account_txns > 1:
        appearances = pd.concat([df["Account"], df["Account.1"]]).value_counts()
        keep = set(appearances[appearances >= min_account_txns].index)
        before = len(df)
        df = df[df["Account"].isin(keep) & df["Account.1"].isin(keep)]
        print(f"    graph pruning: {len(keep):,} accounts with >= "
              f"{min_account_txns} transactions, {len(df):,} of {before:,} rows kept",
              flush=True)

    grouped = df.groupby(["Account", "Account.1"], observed=True).agg(
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


def _prune_to_cycle_candidates(g: nx.DiGraph) -> nx.DiGraph:
    """Repeatedly drop accounts with no inflow or no outflow.

    Such an account cannot sit on a cycle, so removing it is lossless and
    shrinks the search substantially.
    """
    g = g.copy()
    while True:
        dead = [n for n in g if g.in_degree(n) == 0 or g.out_degree(n) == 0]
        if not dead:
            return g
        g.remove_nodes_from(dead)


def _cycle_span_hours(g: nx.DiGraph, cycle: list) -> float:
    """Hours between the earliest and latest activity on a cycle."""
    times = []
    for i, node in enumerate(cycle):
        edge = g.edges[node, cycle[(i + 1) % len(cycle)]]
        times.extend([edge["first"], edge["last"]])
    return (max(times) - min(times)).total_seconds() / 3600


def cycle_features(g: nx.DiGraph, window_hours: int = CYCLE_WINDOW_HOURS,
                   max_length: int = MAX_CYCLE_LENGTH,
                   max_cycles: int | None = None) -> pd.DataFrame:
    """Per account: how many tight cycles, how short, how valuable.

    A cycle only counts if it closes inside `window_hours`. Money returning to
    its origin within a day is layering; the same loop over months is trade.

    `max_cycles` caps enumeration. Cycle counts grow explosively with graph
    size - HI-Small produced 372,952 - and on a larger graph the enumeration
    itself is what exhausts memory. Stopping early biases the counts downward
    but leaves them comparable across accounts, which is what a tree model
    needs; an uncapped run that gets killed produces nothing at all.
    """
    candidates = _prune_to_cycle_candidates(g)
    stats: dict[str, dict] = defaultdict(
        lambda: {"cycles": 0, "tight_cycles": 0, "min_len": np.nan,
                 "min_span": np.nan, "max_value": 0.0})

    seen = 0
    for cycle in nx.simple_cycles(candidates, length_bound=max_length):
        seen += 1
        if max_cycles is not None and seen > max_cycles:
            print(f"    cycle enumeration capped at {max_cycles:,}", flush=True)
            break
        span = _cycle_span_hours(candidates, cycle)
        tight = span <= window_hours
        value = min(candidates.edges[cycle[i], cycle[(i + 1) % len(cycle)]]["total"]
                    for i in range(len(cycle)))
        for node in cycle:
            s = stats[node]
            s["cycles"] += 1
            if tight:
                s["tight_cycles"] += 1
                s["min_len"] = np.nanmin([s["min_len"], len(cycle)])
                s["min_span"] = np.nanmin([s["min_span"], span])
                s["max_value"] = max(s["max_value"], value)

    if not stats:
        return pd.DataFrame(columns=["cycle_count", "tight_cycle_count",
                                     "shortest_cycle", "fastest_cycle_hours",
                                     "largest_cycle_value"])

    out = pd.DataFrame.from_dict(stats, orient="index")
    out.columns = ["cycle_count", "tight_cycle_count", "shortest_cycle",
                   "fastest_cycle_hours", "largest_cycle_value"]
    return out.astype("float32")


def scatter_gather_features(g: nx.DiGraph,
                            window_hours: int = SCATTER_GATHER_WINDOW_HOURS) -> pd.DataFrame:
    """Per account: does money fan out and reconverge, or converge then fan out?

    SCATTER-GATHER is one source paying several intermediaries who all pay one
    destination. GATHER-SCATTER is the reverse. Both are visible two hops from
    the middle account, which is why an aggregate degree count misses them: the
    middle account looks unremarkable on its own.
    """
    rows = {}
    for node in g.nodes:
        successors = list(g.successors(node))
        predecessors = list(g.predecessors(node))
        if not successors and not predecessors:
            continue

        # Scatter-gather: this account pays several accounts that share a payee.
        shared_payees: dict[str, int] = defaultdict(int)
        for mid in successors:
            for payee in g.successors(mid):
                if payee != node:
                    shared_payees[payee] += 1
        sg_width = max(shared_payees.values(), default=0)

        # Gather-scatter: several accounts pay this one, sharing a payer.
        shared_payers: dict[str, int] = defaultdict(int)
        for mid in predecessors:
            for payer in g.predecessors(mid):
                if payer != node:
                    shared_payers[payer] += 1
        gs_width = max(shared_payers.values(), default=0)

        # How fast the fan-out happens. A burst inside the window is the signal;
        # the same spread over weeks is ordinary payment activity.
        out_times = [g.edges[node, s]["first"] for s in successors]
        if len(out_times) > 1:
            burst = (max(out_times) - min(out_times)).total_seconds() / 3600
            fan_out_burst = len(successors) if burst <= window_hours else 0
        else:
            fan_out_burst = 0

        rows[node] = {
            "scatter_gather_width": sg_width,
            "gather_scatter_width": gs_width,
            "two_hop_payees": len(shared_payees),
            "two_hop_payers": len(shared_payers),
            "fan_out_burst": fan_out_burst,
            "out_degree": len(successors),
            "in_degree": len(predecessors),
        }

    return pd.DataFrame.from_dict(rows, orient="index").astype("float32")


def build_pattern_features(df: pd.DataFrame,
                           with_chains: bool = False,
                           min_account_txns: int = 1,
                           max_cycles: int | None = None) -> pd.DataFrame:
    """All pattern counts, one row per account.

    `df` must already carry a `_ts` datetime column and must cover the training
    period only - these are fitted statistics and applying them to rows the
    model will be scored on would leak the future.

    `with_chains` adds the STACK and BIPARTITE detectors from `chains.py`.

    `min_account_txns` and `max_cycles` bound the work for large datasets; both
    are inert at their defaults. See `build_temporal_graph` and `cycle_features`
    for why each is needed above a few million rows.
    """
    g = build_temporal_graph(df, min_account_txns=min_account_txns)
    print(f"    graph: {g.number_of_nodes():,} nodes, "
          f"{g.number_of_edges():,} edges", flush=True)
    cycles = cycle_features(g, max_cycles=max_cycles)
    scatter = scatter_gather_features(g)
    out = scatter.join(cycles, how="outer")

    if with_chains:
        from features.chains import build_chain_features
        chain_feats, _ = build_chain_features(g)
        out = out.join(chain_feats, how="outer")

    # An account absent from the cycle table sits on no cycles - that is a real
    # zero, not a missing value.
    for col in ("cycle_count", "tight_cycle_count", "largest_cycle_value"):
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    return out.astype("float32")
