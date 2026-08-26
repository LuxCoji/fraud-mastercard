"""STACK and BIPARTITE detection - the two shapes the pattern counts only approximate.

Of the eight labelled laundering shapes, five are counted directly by
`graph_patterns.py`. Two are not, and the evidence says they matter:
`from_two_hop_payers` and `to_two_hop_payers` - crude stand-ins for bipartite
structure - are the strongest of the pattern features added so far. If a rough
approximation ranks that highly, the real structure is worth building.

**STACK** is layering: A -> B -> C -> D -> E, each hop passing most of the money
onward. No single account looks unusual; the chain is the evidence. Detected by
walking forward from each account and keeping paths where the value is
conserved and the hops happen in order.

**BIPARTITE** is one group of accounts paying another group, many-to-many, so no
individual account has a striking degree. Detected by finding sets of payers
that share an unusually large set of payees.

This module also returns the chains themselves, not only counts. A score tells
an investigator a transaction is suspicious; a chain tells them what to look at.
That distinction is what turns an alert into a case.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

# A hop "keeps" the money if the onward amount is a similar order of magnitude
# to what arrived. Measured on the 43 labelled STACK rings, the median hop
# ratio is 0.99 - money really is conserved - but the spread is enormous
# (p10 0.05, p90 28.7), because a layering account also mixes in other funds.
# A tight 0.70-1.10 band captured only 34.8% of real hops. This band is wide
# enough to admit them while still excluding a hop that loses or gains an order
# of magnitude, which is a different transaction rather than a continuation.
VALUE_CONSERVATION_LOW = 0.10
VALUE_CONSERVATION_HIGH = 10.0

MIN_CHAIN_LENGTH = 3          # A -> B -> C is the shortest thing worth calling a chain
MAX_CHAIN_LENGTH = 8          # labelled STACK rings carry a median of 10 transactions

# Labelled STACK rings span a median of 73 hours and a p90 of 101. A 72-hour
# window - the obvious round number - excluded more than half of them by an
# hour. 120 covers the p90 with margin.
CHAIN_WINDOW_HOURS = 120

MIN_BIPARTITE_PAYERS = 3
MIN_BIPARTITE_PAYEES = 3


@dataclass
class Chain:
    """One layering path, kept so an investigator can see the flow."""

    accounts: list[str]
    amounts: list[float]
    span_hours: float

    @property
    def length(self) -> int:
        return len(self.accounts) - 1

    @property
    def retained(self) -> float:
        """What fraction of the starting amount survives to the end."""
        return self.amounts[-1] / self.amounts[0] if self.amounts[0] else np.nan

    def describe(self) -> str:
        flow = " -> ".join(self.accounts)
        return (f"{flow}  |  {self.length} hops, "
                f"{self.amounts[0]:,.0f} in / {self.amounts[-1]:,.0f} out "
                f"({self.retained:.0%} retained) over {self.span_hours:.0f}h")


def find_chains(g: nx.DiGraph, max_length: int = MAX_CHAIN_LENGTH,
                window_hours: int = CHAIN_WINDOW_HOURS) -> list[Chain]:
    """Walk forward from every account, keeping value-conserving paths.

    A path continues only while each hop forwards most of what arrived and the
    whole chain stays inside the time window. Both conditions are what separate
    layering from ordinary sequential business.
    """
    chains: list[Chain] = []

    def walk(path: list[str], amounts: list[float], start_time, last_time) -> None:
        if len(path) - 1 >= MIN_CHAIN_LENGTH:
            span = (last_time - start_time).total_seconds() / 3600
            chains.append(Chain(list(path), list(amounts), span))
        if len(path) - 1 >= max_length:
            return

        current, arriving = path[-1], amounts[-1]
        for nxt in g.successors(current):
            if nxt in path:                 # a repeat is a cycle, handled elsewhere
                continue
            edge = g.edges[current, nxt]
            onward = edge["total"]
            ratio = onward / arriving if arriving else 0
            if not (VALUE_CONSERVATION_LOW <= ratio <= VALUE_CONSERVATION_HIGH):
                continue
            if edge["first"] < last_time:   # the next hop must come after this one
                continue
            if (edge["first"] - start_time).total_seconds() / 3600 > window_hours:
                continue
            path.append(nxt); amounts.append(onward)
            walk(path, amounts, start_time, edge["last"])
            path.pop(); amounts.pop()

    for source in g.nodes:
        for first_hop in g.successors(source):
            edge = g.edges[source, first_hop]
            walk([source, first_hop], [edge["total"]], edge["first"], edge["last"])

    return chains


def chain_features(chains: list[Chain]) -> pd.DataFrame:
    """Per account: how deep in a layering chain does it sit?"""
    stats: dict[str, dict] = defaultdict(
        lambda: {"chains": 0, "longest": 0, "deepest_position": 0,
                 "max_value": 0.0, "fastest_hours": np.nan})

    for chain in chains:
        for position, account in enumerate(chain.accounts):
            s = stats[account]
            s["chains"] += 1
            s["longest"] = max(s["longest"], chain.length)
            s["deepest_position"] = max(s["deepest_position"], position)
            s["max_value"] = max(s["max_value"], chain.amounts[0])
            s["fastest_hours"] = np.nanmin([s["fastest_hours"], chain.span_hours])

    if not stats:
        return pd.DataFrame(columns=["chain_count", "longest_chain",
                                     "chain_depth", "chain_max_value",
                                     "chain_fastest_hours"])
    out = pd.DataFrame.from_dict(stats, orient="index")
    out.columns = ["chain_count", "longest_chain", "chain_depth",
                   "chain_max_value", "chain_fastest_hours"]
    return out.astype("float32")


def bipartite_features(g: nx.DiGraph,
                       min_payers: int = MIN_BIPARTITE_PAYERS,
                       min_payees: int = MIN_BIPARTITE_PAYEES) -> pd.DataFrame:
    """Per account: is it part of a group paying another group?

    For each account, look at who else pays its payees. A high overlap means
    this account sits inside a many-to-many block rather than having its own
    distinct set of counterparties - which is what BIPARTITE laundering looks
    like from the inside, and why per-account degree misses it.
    """
    rows: dict[str, dict] = {}

    for node in g.nodes:
        payees = set(g.successors(node))
        if len(payees) < min_payees:
            continue

        # Other accounts paying the same payees.
        peers: dict[str, int] = defaultdict(int)
        for payee in payees:
            for other in g.predecessors(payee):
                if other != node:
                    peers[other] += 1

        strong_peers = [p for p, shared in peers.items() if shared >= min_payees]
        if len(strong_peers) < min_payers:
            continue

        # How much of this account's payee set is shared with its peers?
        shared_payees = set()
        for peer in strong_peers:
            shared_payees |= (payees & set(g.successors(peer)))

        rows[node] = {
            "bipartite_peers": len(strong_peers),
            "bipartite_shared_payees": len(shared_payees),
            "bipartite_density": len(shared_payees) / len(payees),
            "bipartite_block_size": len(strong_peers) + len(payees),
        }

    if not rows:
        return pd.DataFrame(columns=["bipartite_peers", "bipartite_shared_payees",
                                     "bipartite_density", "bipartite_block_size"])
    return pd.DataFrame.from_dict(rows, orient="index").astype("float32")


def build_chain_features(g: nx.DiGraph) -> tuple[pd.DataFrame, list[Chain]]:
    """Chain and bipartite features, plus the chains themselves.

    The chains are returned rather than discarded so that a flagged account can
    be shown as a flow - the difference between an alert and a case.
    """
    chains = find_chains(g)
    features = chain_features(chains).join(bipartite_features(g), how="outer")

    for col in ("chain_count", "longest_chain", "chain_depth", "chain_max_value",
                "bipartite_peers", "bipartite_shared_payees", "bipartite_block_size"):
        if col in features.columns:
            features[col] = features[col].fillna(0.0)
    return features.astype("float32"), chains
