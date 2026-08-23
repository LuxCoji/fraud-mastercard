"""Step 1 gate: prove every raw source loads and report what it contains.

Run: python src/data/verify.py
Nothing downstream is trusted until this passes.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data import loaders


def _report(name: str, df: pd.DataFrame, label: str, time_col: str) -> dict:
    pos = int(df[label].sum())
    n = len(df)
    rate = pos / n
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  rows            {n:>12,}")
    print(f"  columns         {df.shape[1]:>12,}")
    print(f"  label           {label!r}")
    print(f"  positives       {pos:>12,}")
    print(f"  base rate       {rate:>12.4%}")
    print(f"  imbalance       {'1 in ' + format(round(1 / rate), ','):>12}")
    print(f"  time column     {time_col!r}  "
          f"range [{df[time_col].min()}, {df[time_col].max()}]")
    print(f"  monotonic time  {df[time_col].is_monotonic_increasing}")
    print(f"  memory          {df.memory_usage(deep=True).sum() / 1e9:>9.2f} GB")
    return {"source": name, "rows": n, "cols": df.shape[1],
            "positives": pos, "base_rate": rate}


def main() -> None:
    rows = []

    # --- card ---------------------------------------------------------------
    df = loaders.load_card()
    r = _report("CARD  (IEEE-CIS, card-not-present)", df, "isFraud", "TransactionDT")
    v_cols = [c for c in df.columns if c.startswith("V")]
    d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
    id_cols = [c for c in df.columns if c.startswith("id_")]
    ident_cov = df[id_cols[0]].notna().mean() if id_cols else 0.0
    print(f"  V columns       {len(v_cols):>12}   <-- dropped by the last attempt")
    print(f"  D columns       {len(d_cols):>12}")
    print(f"  id_ columns     {len(id_cols):>12}   coverage {ident_cov:.1%}")
    print(f"  card1 cardinality {df['card1'].nunique():>10,}")
    r["note"] = f"{len(v_cols)} V-cols recovered"
    rows.append(r)
    del df; gc.collect()

    # --- transfer -----------------------------------------------------------
    df = loaders.load_transfer()
    r = _report("TRANSFER  (PaySim, P2P push)", df, "isFraud", "step")
    print(f"  txn types       {dict(df['type'].value_counts())}")
    fraud_types = df.loc[df.isFraud == 1, "type"].value_counts()
    print(f"  fraud only in   {dict(fraud_types)}")
    r["note"] = "fraud confined to TRANSFER/CASH_OUT"
    rows.append(r)
    del df; gc.collect()

    # --- wire ---------------------------------------------------------------
    df = loaders.load_wire()
    label = cfg.LABEL_COLUMN["ibm_aml"]
    r = _report("WIRE  (IBM AML HI-Small)", df, label, "Timestamp")
    n_accounts = pd.concat([df.iloc[:, 1], df.iloc[:, 3]]).nunique()
    print(f"  distinct accounts {n_accounts:>10,}")
    with open(cfg.AML_PATTERNS) as fh:
        pattern_lines = [ln for ln in fh if ln.startswith("BEGIN LAUNDERING")]
    print(f"  labelled rings  {len(pattern_lines):>12}   <-- graph ground truth")
    r["note"] = f"{len(pattern_lines)} labelled ring topologies"
    rows.append(r)
    del df; gc.collect()

    # --- summary ------------------------------------------------------------
    out = pd.DataFrame(rows)
    print(f"\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
    print(out.to_string(index=False))
    dest = cfg.REPORTS / "step1_data_verification.csv"
    out.to_csv(dest, index=False)
    print(f"\nwritten -> {dest}")
    spread = out.base_rate.max() / out.base_rate.min()
    print(f"\nbase-rate spread across rails: {spread:.1f}x")
    print("-> per-rail thresholds are mandatory; a single global threshold")
    print("   would give three very different false-positive rates.")


if __name__ == "__main__":
    main()
