"""Raw loaders, one per rail.

Each loader returns the raw frame with its native columns intact. No canonical
schema, no cross-source truncation: the previous attempt at this project mapped
all three sources into shared feature slots and dropped IEEE-CIS's V1-V339 in
the process, which is where that dataset's signal lives. Each rail keeps its
own columns and its own model.

Memory note: this machine has ~6 GB free and IEEE-CIS is 1.65 GB as float64
before pandas even finishes consolidating blocks. Every loader therefore forces
narrow dtypes *at read time* rather than downcasting afterwards, and caches the
result as Parquet so the CSV parse is paid exactly once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

_SAMPLE_ROWS = 20_000


def _narrow_dtypes(path: Path, keep_int: tuple[str, ...] = ()) -> dict:
    """Infer a read-time dtype map that forces every numeric column to float32.

    float32 is NaN-capable, so integer-looking columns that turn out to have
    missing values further down the file do not force a silent widening back to
    float64. Columns in `keep_int` are pinned to int32 instead.
    """
    sample = pd.read_csv(path, nrows=_SAMPLE_ROWS)
    dtypes: dict[str, str] = {}
    for col, dt in sample.dtypes.items():
        if col in keep_int:
            dtypes[col] = "int32"
        elif pd.api.types.is_numeric_dtype(dt):
            dtypes[col] = "float32"
    return dtypes


def _cached(name: str, build) -> pd.DataFrame:
    """Read from the Parquet cache, building it on first call."""
    cache = cfg.INTERIM / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = build()
    df.to_parquet(cache, index=False, compression="snappy")
    return df


def load_card(nrows: int | None = None, with_identity: bool = True) -> pd.DataFrame:
    """IEEE-CIS card-not-present authorisations.

    `TransactionDT` is a seconds offset from an arbitrary origin, not a
    timestamp. It is monotonic, which is all the chronological split needs.
    Identity is a left join: only ~24% of transactions carry it, and that
    missingness is itself informative rather than a defect to impute away.
    """
    def build() -> pd.DataFrame:
        keep = ("TransactionID", "TransactionDT", "isFraud")
        txn = pd.read_csv(
            cfg.IEEE_TRANSACTION, nrows=nrows,
            dtype=_narrow_dtypes(cfg.IEEE_TRANSACTION, keep_int=keep),
        )
        if with_identity:
            ident = pd.read_csv(
                cfg.IEEE_IDENTITY,
                dtype=_narrow_dtypes(cfg.IEEE_IDENTITY, keep_int=("TransactionID",)),
            )
            txn = txn.merge(ident, on="TransactionID", how="left")
        return txn

    if nrows is not None or not with_identity:
        return build()
    return _cached("card", build)


def load_transfer(nrows: int | None = None) -> pd.DataFrame:
    """PaySim P2P mobile-money transfers.

    `step` is an hour index (1 step = 1 hour, 744 steps = 30 days).
    `isFlaggedFraud` is the simulator's own naive rule, not a label - it must
    never be used as a feature or it leaks the answer.
    """
    def build() -> pd.DataFrame:
        keep = ("step", "isFraud", "isFlaggedFraud")
        return pd.read_csv(
            cfg.PAYSIM, nrows=nrows,
            dtype=_narrow_dtypes(cfg.PAYSIM, keep_int=keep),
        )

    return build() if nrows is not None else _cached("transfer", build)


def load_wire(nrows: int | None = None) -> pd.DataFrame:
    """IBM AML interbank wires (HI-Small).

    Laundering is a topology, not a row property, so this frame is the input to
    the graph builder rather than to a row-level model directly.
    """
    def build() -> pd.DataFrame:
        return pd.read_csv(
            cfg.AML_TRANS, nrows=nrows,
            dtype=_narrow_dtypes(cfg.AML_TRANS, keep_int=("Is Laundering",)),
        )

    return build() if nrows is not None else _cached("wire", build)


LOADERS = {"card": load_card, "transfer": load_transfer, "wire": load_wire}
