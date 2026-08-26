"""Parse the 370 labelled laundering rings from HI-Small_Patterns.txt.

The transaction file tells you *which rows* are laundering. This file tells you
which rows belong to the *same* laundering operation, and what shape it is. That
distinction is the whole point of the wire rail: catching one leg of an
eight-hop cycle is not the same as catching the cycle.

File format:

    BEGIN LAUNDERING ATTEMPT - CYCLE: Max 4-Cycle
    2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,...,ACH,1
    ... more transactions ...
    END LAUNDERING ATTEMPT - CYCLE

Each block is one ring. The eight shapes are CYCLE, FAN-IN, FAN-OUT,
GATHER-SCATTER, SCATTER-GATHER, BIPARTITE, STACK and RANDOM.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

BEGIN = "BEGIN LAUNDERING ATTEMPT"
END = "END LAUNDERING ATTEMPT"


@dataclass
class Ring:
    """One labelled laundering operation."""

    ring_id: int
    pattern: str
    transactions: list[dict] = field(default_factory=list)

    @property
    def accounts(self) -> set[str]:
        out = set()
        for t in self.transactions:
            out.add(t["from_account"])
            out.add(t["to_account"])
        return out


def _parse_pattern(line: str) -> str:
    """Pull the shape name out of a BEGIN line.

    The file is inconsistent: some lines read
    'BEGIN LAUNDERING ATTEMPT - CYCLE: Max 4-Cycle' and others put the shape
    after a second dash. Matching on the known vocabulary avoids guessing.
    """
    known = ["GATHER-SCATTER", "SCATTER-GATHER", "FAN-OUT", "FAN-IN",
             "BIPARTITE", "CYCLE", "STACK", "RANDOM"]
    upper = line.upper()
    for name in known:
        if name in upper.replace(BEGIN, ""):
            return name
    return "UNKNOWN"


def load_rings(path: Path | None = None) -> list[Ring]:
    """Read every labelled ring."""
    path = path or cfg.AML_PATTERNS
    rings: list[Ring] = []
    current: Ring | None = None

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(BEGIN):
                current = Ring(ring_id=len(rings), pattern=_parse_pattern(line))
            elif line.startswith(END):
                if current and current.transactions:
                    rings.append(current)
                current = None
            elif current is not None:
                parts = line.split(",")
                if len(parts) < 11:
                    continue
                current.transactions.append({
                    "timestamp": parts[0],
                    "from_bank": parts[1],
                    "from_account": parts[2],
                    "to_bank": parts[3],
                    "to_account": parts[4],
                    "amount_received": float(parts[5]),
                    "amount_paid": float(parts[7]),
                    "payment_format": parts[9],
                })
    return rings


def rings_to_frame(rings: list[Ring]) -> pd.DataFrame:
    """One row per ring transaction, tagged with its ring id and shape.

    This is the join key back to the transaction file: a row in HI-Small_Trans
    belongs to a ring if (timestamp, from_account, to_account, amount) matches.
    """
    rows = []
    for ring in rings:
        for t in ring.transactions:
            rows.append({"ring_id": ring.ring_id, "pattern": ring.pattern, **t})
    return pd.DataFrame(rows)


def summarise(rings: list[Ring]) -> pd.DataFrame:
    """Size and shape of each ring type."""
    rows = []
    for ring in rings:
        rows.append({
            "pattern": ring.pattern,
            "transactions": len(ring.transactions),
            "accounts": len(ring.accounts),
        })
    df = pd.DataFrame(rows)
    return df.groupby("pattern").agg(
        rings=("transactions", "size"),
        total_txns=("transactions", "sum"),
        median_txns=("transactions", "median"),
        median_accounts=("accounts", "median"),
    ).sort_values("rings", ascending=False)


if __name__ == "__main__":
    rings = load_rings()
    print(f"parsed {len(rings)} rings, "
          f"{sum(len(r.transactions) for r in rings):,} transactions\n")
    print(summarise(rings).to_string())
