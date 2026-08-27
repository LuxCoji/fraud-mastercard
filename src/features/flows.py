"""Trace money flows by making transactions the nodes, not accounts.

The account graph collapses every payment between A and B into a single edge
carrying summed totals and a first/last timestamp. That is fine for asking "how
many counterparties does this account have" and useless for asking "did this
money move onward" - the sequence is exactly what got summed away.

Measured consequence: the account-graph chain detector found **7 of 43** labelled
STACK rings and **4 of 49** BIPARTITE, and accidentally found cycles better than
the chains it was written for.

Here a node is one transaction and an edge means *this money could have funded
that payment*:

    txn_i (A -> B at t_i)   ->   txn_j (B -> C at t_j)

    when   t_j > t_i        the onward payment comes after the money arrived
    and    t_j - t_i <= W   it moved on promptly rather than months later

Two properties fall out of that definition, and both matter:

**The graph is acyclic.** Time strictly increases along every edge, so path
enumeration cannot loop and is far cheaper than on an account graph - which is
what exhausted the budget at 1.2M cycles on HI-Medium.

**A laundering cycle is a path, not a cycle.** A -> B -> C -> A is three
transactions in time order whose last receiver is the first sender. So one
traversal finds layering chains and circular flows together, and tells them
apart by comparing endpoints.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# How long money may sit before an onward payment still counts as the same flow.
# Labelled STACK rings span a median of 73 hours and a p90 of 101, so 120 covers
# the p90 with margin. A 72-hour window - the obvious round number - excluded
# more than half of them by a single hour.
FLOW_WINDOW_HOURS = 120

# An onward payment counts as carrying the same money if it is within this band
# of what arrived. Measured on the labelled rings the median hop ratio is 0.99 -
# value really is conserved - but the spread is enormous (p10 0.05, p90 28.7),
# because a laundering account mixes in other funds. A tight band captured only
# 34.8% of genuine hops.
VALUE_LOW, VALUE_HIGH = 0.10, 10.0

MIN_FLOW_HOPS = 2          # A -> B -> C is the shortest thing worth showing
MAX_FLOW_HOPS = 8

# Onward edges kept per transaction. Without this the graph is unbounded: on the
# training period it reached **41.9 million edges** and the run was OOM-killed.
# An account that pays fifty counterparties after receiving does not need fifty
# traced continuations - the fan features already measure that width, and what
# the tracer is for is depth. Keeping the earliest few preserves the chain.
MAX_FOLLOWERS = 6


@dataclass
class Flow:
    """One traced path of money through several accounts."""

    transactions: list[int]           # row positions, in time order
    accounts: list[str]               # sender of the first, then each receiver
    amounts: list[float]
    times: list[pd.Timestamp]

    @property
    def hops(self) -> int:
        return len(self.transactions)

    @property
    def span_hours(self) -> float:
        return (self.times[-1] - self.times[0]).total_seconds() / 3600

    @property
    def is_cycle(self) -> bool:
        """Did the money come back to where it started?"""
        return self.accounts[0] == self.accounts[-1]

    @property
    def retained(self) -> float:
        return self.amounts[-1] / self.amounts[0] if self.amounts[0] else np.nan

    def describe(self) -> str:
        shape = "CYCLE" if self.is_cycle else "CHAIN"
        path = " -> ".join(self.accounts)
        return (f"{shape} {self.hops} hops | {path} | "
                f"{self.amounts[0]:,.0f} in / {self.amounts[-1]:,.0f} out "
                f"({self.retained:.0%}) over {self.span_hours:.0f}h")


def build_flow_graph(txns: pd.DataFrame,
                     sender: str = "Account", receiver: str = "Account.1",
                     amount: str = "Amount Paid", time: str = "_ts",
                     window_hours: int = FLOW_WINDOW_HOURS,
                     max_followers: int = MAX_FOLLOWERS) -> dict[int, list[int]]:
    """Adjacency from each transaction to the payments its money could have funded.

    Returned as a plain dict of row position -> row positions rather than a
    networkx object: the traversal below is the only consumer, and the dict is
    both faster to build and cheaper to hold.
    """
    order = np.argsort(txns[time].values, kind="stable")
    ordered = txns.iloc[order]

    times = ordered[time].values
    senders = ordered[sender].to_numpy()
    receivers = ordered[receiver].to_numpy()
    amounts = ordered[amount].to_numpy(dtype="float64")

    # For each account, the positions where it *sends*, already in time order -
    # so the onward payments after a given moment are a contiguous slice.
    sends_by_account: dict[str, list[int]] = defaultdict(list)
    for position, account in enumerate(senders):
        sends_by_account[account].append(position)

    window = np.timedelta64(window_hours, "h")
    adjacency: dict[int, list[int]] = {}

    for i in range(len(ordered)):
        onward = sends_by_account.get(receivers[i])
        if not onward:
            continue

        # The receiver's own sends, restricted to after this transaction.
        send_times = [times[j] for j in onward]
        start = bisect_right(send_times, times[i])
        deadline = times[i] + window

        followers = []
        for j in onward[start:]:
            if times[j] > deadline:
                break
            if senders[j] == receivers[j]:          # self-transfer, not a hop
                continue
            ratio = amounts[j] / amounts[i] if amounts[i] else 0.0
            if VALUE_LOW <= ratio <= VALUE_HIGH:
                followers.append(j)
                if len(followers) >= max_followers:
                    break
        if followers:
            adjacency[i] = followers

    # Keep the mapping back to the caller's row positions.
    adjacency["_order"] = order
    return adjacency


def find_flows(txns: pd.DataFrame, adjacency: dict,
               min_hops: int = MIN_FLOW_HOPS,
               max_hops: int = MAX_FLOW_HOPS,
               max_flows: int = 10_000) -> list[Flow]:
    """Walk every path through the flow graph.

    Bounded by `max_flows` because a dense component can produce very many
    paths, and an investigator reads the first few. The graph is acyclic by
    construction, so this terminates without needing a visited set.
    """
    order = adjacency["_order"]
    ordered = txns.iloc[order]
    senders = ordered.iloc[:, ordered.columns.get_loc("Account")].to_numpy()
    receivers = ordered.iloc[:, ordered.columns.get_loc("Account.1")].to_numpy()
    amounts = ordered["Amount Paid"].to_numpy(dtype="float64")
    times = ordered["_ts"].to_numpy()

    flows: list[Flow] = []

    def walk(path: list[int], visited: set) -> None:
        if len(flows) >= max_flows:
            return
        if len(path) >= min_hops:
            flows.append(Flow(
                transactions=[int(order[p]) for p in path],
                accounts=[senders[path[0]]] + [receivers[p] for p in path],
                amounts=[amounts[p] for p in path],
                times=[pd.Timestamp(times[p]) for p in path],
            ))
        if len(path) >= max_hops:
            return
        for nxt in adjacency.get(path[-1], ()):
            # An account may appear at most twice in a flow: once on the way out
            # and once on the way back, which is what closes a cycle. Without
            # this the traversal ping-pongs between a pair of accounts and fills
            # the entire budget with variations of one path - measured, it
            # produced 50,000 flows that between them touched a handful of
            # accounts and reached fewer labelled rings than the account-graph
            # version it replaced.
            account = receivers[nxt]
            if visited.get(account, 0) >= 2:
                continue
            visited[account] = visited.get(account, 0) + 1
            path.append(nxt)
            walk(path, visited)
            path.pop()
            visited[account] -= 1

    # Start from transactions nothing feeds into. Every flow has to begin
    # somewhere, and starting mid-path only re-discovers suffixes of paths the
    # sources already produce.
    has_predecessor = set()
    for source, targets in adjacency.items():
        if source != "_order":
            has_predecessor.update(targets)

    starts = [k for k in adjacency if k != "_order" and k not in has_predecessor]
    starts += [k for k in adjacency if k != "_order" and k in has_predecessor]

    for start in starts:
        walk([start], {receivers[start]: 1, senders[start]: 1})
        if len(flows) >= max_flows:
            break

    return flows


def summarise(flows: list[Flow]) -> pd.DataFrame:
    """One row per flow, for a quick look at what was found."""
    return pd.DataFrame([{
        "hops": f.hops, "shape": "cycle" if f.is_cycle else "chain",
        "accounts": len(set(f.accounts)), "span_hours": f.span_hours,
        "amount_in": f.amounts[0], "amount_out": f.amounts[-1],
        "retained": f.retained,
    } for f in flows])


def flow_features(txns: pd.DataFrame, adjacency: dict, flows: list[Flow]) -> pd.DataFrame:
    """Per account: what do the traced flows say about it?

    The tracer reaches **79% of the labelled rings** against 19% for the
    account-graph detectors it replaced, but it was only ever wired into case
    display - the model has never seen any of it. These are the same paths,
    summarised per account so a gradient-boosted model can use them.

    Position in a flow is the point. An account that only ever *starts* flows is
    a source; one that only ever ends them is a destination; one that sits in the
    middle of long fast flows is a conduit, and that is what layering looks like
    from the outside. A degree count cannot express any of that.
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "flows": 0, "cycles": 0, "longest": 0, "deepest": 0,
        "fastest": np.nan, "max_value": 0.0,
        "as_source": 0, "as_middle": 0, "as_sink": 0,
    })

    for flow in flows:
        last = len(flow.accounts) - 1
        for position, account in enumerate(flow.accounts):
            s = stats[account]
            s["flows"] += 1
            s["cycles"] += int(flow.is_cycle)
            s["longest"] = max(s["longest"], flow.hops)
            s["deepest"] = max(s["deepest"], position)
            s["fastest"] = np.nanmin([s["fastest"], flow.span_hours])
            s["max_value"] = max(s["max_value"], flow.amounts[0])
            if position == 0:
                s["as_source"] += 1
            elif position == last:
                s["as_sink"] += 1
            else:
                s["as_middle"] += 1

    columns = ["flow_count", "flow_cycles", "flow_longest", "flow_depth",
               "flow_fastest_hours", "flow_max_value",
               "flow_as_source", "flow_as_middle", "flow_as_sink"]
    if not stats:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame.from_dict(stats, orient="index")
    out.columns = columns

    # The shape of an account's participation, not just how much of it. A
    # conduit sits in the middle; a mule collects at the end.
    total = out.flow_count.replace(0, np.nan)
    out["flow_middle_ratio"] = out.flow_as_middle / total
    out["flow_cycle_ratio"] = out.flow_cycles / total
    return out.astype("float32")


def build_flow_account_features(df: pd.DataFrame, max_flows: int = 2_000_000,
                                window_hours: int = FLOW_WINDOW_HOURS,
                                max_followers: int = MAX_FOLLOWERS) -> pd.DataFrame:
    """Trace the flows and summarise them per account, in one call.

    Flows are accumulated as they are found rather than collected into a list.
    Materialising 400,000 `Flow` objects - each carrying its accounts, amounts
    and timestamps - was what exhausted memory; the statistics need only one
    flow at a time, so nothing is held.

    `df` must carry `_ts` and must cover the training period only - these are
    fitted statistics and letting them see the scoring period would leak.
    """
    adjacency = build_flow_graph(df, window_hours=window_hours,
                                 max_followers=max_followers)
    edges = sum(len(v) for k, v in adjacency.items() if k != "_order")

    order = adjacency["_order"]
    ordered = df.iloc[order]
    senders = ordered["Account"].to_numpy()
    receivers = ordered["Account.1"].to_numpy()
    amounts = ordered["Amount Paid"].to_numpy(dtype="float64")
    times = ordered["_ts"].to_numpy()

    stats: dict[str, dict] = defaultdict(lambda: {
        "flows": 0, "cycles": 0, "longest": 0, "deepest": 0,
        "fastest": np.nan, "max_value": 0.0,
        "as_source": 0, "as_middle": 0, "as_sink": 0,
    })
    found = 0

    def record(path: list[int]) -> None:
        nonlocal found
        found += 1
        accounts = [senders[path[0]]] + [receivers[p] for p in path]
        span = (times[path[-1]] - times[path[0]]) / np.timedelta64(1, "h")
        is_cycle = accounts[0] == accounts[-1]
        last = len(accounts) - 1
        start_amount = amounts[path[0]]
        for position, account in enumerate(accounts):
            s = stats[account]
            s["flows"] += 1
            s["cycles"] += is_cycle
            s["longest"] = max(s["longest"], len(path))
            s["deepest"] = max(s["deepest"], position)
            s["fastest"] = np.nanmin([s["fastest"], span])
            s["max_value"] = max(s["max_value"], start_amount)
            if position == 0:
                s["as_source"] += 1
            elif position == last:
                s["as_sink"] += 1
            else:
                s["as_middle"] += 1

    def walk(path: list[int], visited: dict) -> None:
        if found >= max_flows:
            return
        if len(path) >= MIN_FLOW_HOPS:
            record(path)
        if len(path) >= MAX_FLOW_HOPS:
            return
        for nxt in adjacency.get(path[-1], ()):
            account = receivers[nxt]
            if visited.get(account, 0) >= 2:
                continue
            visited[account] = visited.get(account, 0) + 1
            path.append(nxt)
            walk(path, visited)
            path.pop()
            visited[account] -= 1

    has_predecessor = set()
    for source, targets in adjacency.items():
        if source != "_order":
            has_predecessor.update(targets)
    starts = [k for k in adjacency if k != "_order" and k not in has_predecessor]
    starts += [k for k in adjacency if k != "_order" and k in has_predecessor]

    for start in starts:
        walk([start], {receivers[start]: 1, senders[start]: 1})
        if found >= max_flows:
            break

    print(f"    flow tracer: {found:,} flows over {edges:,} edges, "
          f"{len(stats):,} accounts touched", flush=True)

    columns = ["flow_count", "flow_cycles", "flow_longest", "flow_depth",
               "flow_fastest_hours", "flow_max_value",
               "flow_as_source", "flow_as_middle", "flow_as_sink"]
    if not stats:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame.from_dict(stats, orient="index")
    out.columns = columns
    total = out.flow_count.replace(0, np.nan)
    out["flow_middle_ratio"] = out.flow_as_middle / total
    out["flow_cycle_ratio"] = out.flow_cycles / total
    return out.astype("float32")
