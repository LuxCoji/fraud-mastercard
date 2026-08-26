"""Turn wire alerts into investigable cases.

A score tells an investigator that a transaction is suspicious. It does not tell
them what to look at, and 200 disconnected alerts are not 200 pieces of work -
they are one unreadable queue.

This module closes that gap. The model still decides *what* is risky; the graph
explains *why*, by attaching the structure the risky account sits inside:

    transaction -> model -> risk 0.87
                              |
                     graph lookup on the account
                              |
        "8-hop chain A -> B -> C ... , 49,168 in / 37,845 out over 37h"

The model is not replaced and nothing here re-scores anything. The graph search
runs only on accounts the model already flagged, so its cost falls on a few
hundred alerts rather than five million transactions - which is also why it can
afford to be slower than the critical path allows.

Alerts that share an account are merged, so a ring producing twelve alerts
becomes one case with twelve transactions rather than twelve cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from features.chains import Chain, find_chains


@dataclass
class Case:
    """One investigable unit: linked alerts plus the structure behind them."""

    case_id: int
    accounts: set[str]
    transactions: pd.DataFrame
    chains: list[Chain] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)

    @property
    def max_score(self) -> float:
        return float(self.transactions["score"].max())

    @property
    def total_value(self) -> float:
        return float(self.transactions["Amount Paid"].sum())

    @property
    def evidence(self) -> str:
        """What shape, if any, explains this case."""
        if self.cycles:
            shortest = min(len(c) for c in self.cycles)
            return f"CYCLE ({shortest} hops)"
        if self.chains:
            longest = max(c.length for c in self.chains)
            return f"CHAIN ({longest} hops)"
        return "no structure found"

    def summary(self) -> str:
        return (f"case {self.case_id:>3} | {len(self.transactions):>3} alerts | "
                f"{len(self.accounts):>3} accounts | "
                f"{self.total_value:>14,.0f} | "
                f"score {self.max_score:.3f} | {self.evidence}")


def _link_alerts(alerts: pd.DataFrame, sender: str, receiver: str) -> list[set[str]]:
    """Group alerts that share an account into connected components.

    Two alerts touching the same account are almost certainly the same
    operation, so they belong in one case.
    """
    # zip over the two columns rather than itertuples: IBM AML's receiver column
    # is literally named "Account.1", and itertuples mangles a name with a dot
    # into an inaccessible positional attribute.
    linker = nx.Graph()
    linker.add_edges_from(zip(alerts[sender].values, alerts[receiver].values))
    return list(nx.connected_components(linker))


def build_cases(alerts: pd.DataFrame, graph: nx.DiGraph,
                sender: str = "Account", receiver: str = "Account.1",
                max_accounts_per_case: int = 200) -> list[Case]:
    """Group scored alerts into cases and attach the structure behind each.

    `alerts` is the flagged subset with a `score` column. `graph` is the account
    graph built from the same period.

    Cases above `max_accounts_per_case` skip the structure search: a component
    that large is usually a hub rather than a ring, and enumerating its paths
    costs far more than the answer is worth.
    """
    cases: list[Case] = []

    for case_id, accounts in enumerate(_link_alerts(alerts, sender, receiver)):
        mask = alerts[sender].isin(accounts) | alerts[receiver].isin(accounts)
        case = Case(case_id=case_id, accounts=set(accounts),
                    transactions=alerts[mask])

        if len(accounts) <= max_accounts_per_case:
            sub = graph.subgraph(accounts)
            if sub.number_of_edges():
                # A one-node "cycle" is an account paying itself. IBM AML's
                # Reinvestment payment format does exactly that and it is
                # ordinary, so a self-loop is not evidence of anything.
                case.cycles = [c for c in nx.simple_cycles(sub, length_bound=6)
                               if len(c) >= 2]
                case.chains = find_chains(sub)

        cases.append(case)

    # Highest-scoring first: an investigator works from the top.
    return sorted(cases, key=lambda c: -c.max_score)


def format_queue(cases: list[Case], limit: int = 20) -> str:
    """The investigator's queue, most urgent first."""
    lines = [f"{len(cases)} cases from "
             f"{sum(len(c.transactions) for c in cases)} alerts", ""]
    for case in cases[:limit]:
        lines.append(case.summary())
    if len(cases) > limit:
        lines.append(f"... and {len(cases) - limit} more")
    return "\n".join(lines)


def case_detail(case: Case, limit: int = 5) -> str:
    """Everything an investigator needs for one case."""
    lines = [
        f"CASE {case.case_id}",
        f"  alerts        {len(case.transactions)}",
        f"  accounts      {len(case.accounts)}",
        f"  total value   {case.total_value:,.2f}",
        f"  highest score {case.max_score:.4f}",
        f"  evidence      {case.evidence}",
    ]
    if case.cycles:
        lines.append(f"\n  cycles ({len(case.cycles)}):")
        for cycle in sorted(case.cycles, key=len)[:limit]:
            lines.append("    " + " -> ".join(cycle) + f" -> {cycle[0]}")
    if case.chains:
        lines.append(f"\n  chains ({len(case.chains)}):")
        for chain in sorted(case.chains, key=lambda c: -c.length)[:limit]:
            lines.append("    " + chain.describe())
    return "\n".join(lines)
