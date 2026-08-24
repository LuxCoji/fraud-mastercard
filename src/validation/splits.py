"""Train/test splitting for the card rail.

We split by time: train on the earliest transactions, test on the latest.
That is what production looks like - you fit a model on the past and score
the future.

A note on entity overlap, because it is easy to get backwards. The same card
appearing in both train and test is NOT leakage. In production you genuinely do
know a card's history. Leakage is using a card's *future* to score its *past*,
which is a feature-building mistake, not a splitting one.

We still measure the overlap, because it tells you what a score means: a model
tested mostly on cards it has already seen is answering an easier question than
one tested on strangers. Both numbers are worth having.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass
class Split:
    """Row positions for one chronological split."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    cut_time: float

    def __len__(self) -> int:
        return len(self.train_idx) + len(self.test_idx)


def build_uid(df: pd.DataFrame) -> pd.Series:
    """Approximate a physical card: card1 + addr1 + inferred registration day.

    `D1` is days elapsed since the card began, so `day - D1` is the day the card
    was registered - a value that stays constant across all of that card's
    transactions. Combining it with card1 and addr1 gives the UID the reference
    notebooks call the "magic feature".

    Used here for diagnostics and, later, for building history features. It is
    never fed to the model directly.
    """
    day = df["TransactionDT"] / SECONDS_PER_DAY
    registration_day = np.floor(day - df["D1"])
    return (
        df["card1"].astype(str)
        + "_" + df["addr1"].astype(str)
        + "_" + registration_day.astype(str)
    )


def chronological_split(df: pd.DataFrame, time_col: str = "TransactionDT",
                        test_fraction: float = 0.20) -> Split:
    """Hold out the most recent `test_fraction` of transactions.

    The frame is sorted by time first, so this is safe on sources whose rows do
    not arrive in order (IBM AML's timestamps, for example, are not monotonic).
    """
    order = np.argsort(df[time_col].values, kind="stable")
    cut = int(len(order) * (1 - test_fraction))
    return Split(
        train_idx=order[:cut],
        test_idx=order[cut:],
        cut_time=float(df[time_col].values[order[cut]]),
    )


def describe_split(df: pd.DataFrame, split: Split, label_col: str,
                   uid: pd.Series | None = None) -> pd.DataFrame:
    """Summarise a split: size, fraud rate, time range, and entity overlap.

    Printing this before training is the cheapest guard against a silently
    broken split - a test fraud rate far from the train rate usually means the
    sort key was wrong.
    """
    rows = []
    for name, idx in (("train", split.train_idx), ("test", split.test_idx)):
        part = df.iloc[idx]
        rows.append({
            "part": name,
            "rows": len(part),
            "fraud": int(part[label_col].sum()),
            "fraud_rate": part[label_col].mean(),
            "time_min": part["TransactionDT"].min(),
            "time_max": part["TransactionDT"].max(),
        })
    out = pd.DataFrame(rows)

    if uid is not None:
        train_uids = set(uid.iloc[split.train_idx])
        test_uids = uid.iloc[split.test_idx]
        seen = test_uids.isin(train_uids)
        out.attrs["test_uid_seen_in_train"] = float(seen.mean())
        out.attrs["test_uids_total"] = int(test_uids.nunique())
    return out
