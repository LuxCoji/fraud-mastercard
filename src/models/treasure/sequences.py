"""Turn IEEE-CIS into per-card transaction sequences for TREASURE.

TREASURE reads a cardholder's *trajectory*, not a single row. The paper's own
framing: a transaction that looks ordinary in isolation may be the third of
five rapid tests on a stolen card, and only the sequence shows that.

The reimplementation supplied as MCB was rejected because its data generator
wrote the label - `data_gen.py` injects attacks with fixed signatures and marks
exactly those rows, so the model recovers its own if-statements. **The
architecture was never the problem.** This module puts the same architecture on
real, labelled, adversarially-generated data instead.

## The mapping

The paper uses 5 static, 16 dynamic and 2 signal attributes on proprietary Visa
data whose names it cannot disclose. IEEE-CIS is public, so the mapping can be
stated exactly:

| Paper | IEEE-CIS |
| --- | --- |
| static (per-card, constant) | card1-6, addr1-2 |
| dynamic numerical | TransactionAmt, time since last txn, dist1, C1-C14, D1-D15 |
| dynamic categorical | ProductCD, emaildomains, DeviceType, hour, weekday, M1-M9 |
| network signal | isFraud |

## The constraint this data imposes

The median card here has **one** transaction. Cards with three or more cover
66.6% of traffic and carry more fraud (3.92% against 2.65%), so there is a real
sequence signal to find - but a third of the book has no history at all, and on
those rows a sequence model has nothing to read that a per-row model does not
already see.

TREASURE was trained on **six billion** transactions from 30 million
cardholders. IEEE-CIS has 590,540 from 217,850. That is four orders of magnitude
less, against an architecture whose own paper reports that performance improves
with scale. This experiment is therefore a fair test of the architecture *on the
data we have*, not a reproduction of the paper's result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# p99 of transactions per card is 20, so this covers essentially every card
# while keeping the attention matrix small. The paper caps at 512, which is
# sized for cardholders with years of history.
MAX_SEQ_LEN = 32

STATIC_CAT = ["card1", "card2", "card3", "card5", "card4", "card6", "addr1", "addr2"]
DYNAMIC_CAT = ["ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType",
               "hour_of_day", "day_of_week"] + [f"M{i}" for i in range(1, 10)]
DYNAMIC_NUM = (["TransactionAmt", "time_delta", "dist1", "dist2"]
               + [f"C{i}" for i in range(1, 15)]
               + [f"D{i}" for i in range(1, 16)])


def build_uid(df: pd.DataFrame) -> pd.Series:
    """The same card identity the GBDT uses - card1 + addr1 + first-seen day.

    D1 is days since the card was first used, so `day - D1` is the day it
    appeared. Two cards sharing card1 and addr1 but appearing on different days
    are different cards.
    """
    day = df["TransactionDT"] / 86_400
    return (df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_"
            + np.floor(day - df["D1"]).astype(str))


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame,
                        columns: list[str]) -> tuple[dict, dict]:
    """Index every categorical from the training rows only.

    Index 0 is reserved for unseen-or-missing. A category that appears for the
    first time in the test period must not silently take another category's
    embedding, which is what happens if the mapping is fitted on both.
    """
    codes, cardinalities = {}, {}
    for col in columns:
        values = train[col].dropna().unique()
        codes[col] = {v: i + 1 for i, v in enumerate(sorted(values, key=str))}
        cardinalities[col] = len(values) + 1
    return codes, cardinalities


def apply_codes(df: pd.DataFrame, columns: list[str], codes: dict) -> np.ndarray:
    out = np.zeros((len(df), len(columns)), dtype="int32")
    for j, col in enumerate(columns):
        out[:, j] = df[col].map(codes[col]).fillna(0).astype("int32").values
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived fields the sequence needs, in place of a join."""
    out = df.copy()
    out["uid"] = build_uid(out)
    out["hour_of_day"] = ((out["TransactionDT"] // 3600) % 24).astype("int16")
    out["day_of_week"] = ((out["TransactionDT"] // 86_400) % 7).astype("int16")

    # Seconds since this card's previous transaction. The first transaction on a
    # card has no predecessor; -1 marks that rather than 0, which would read as
    # "instantaneous".
    out = out.sort_values(["uid", "TransactionDT"], kind="stable")
    delta = out.groupby("uid")["TransactionDT"].diff()
    out["time_delta"] = delta.fillna(-1).astype("float32")

    # The index MUST be reset here. `build_sequences` records each row's index
    # label so scores can be matched back to labels, but the caller reads those
    # back positionally with `.to_numpy()[row_index]`. After a sort, labels and
    # positions differ, so every score is attached to a different transaction's
    # label - which produced AUC 0.5007 across three seeds, exactly chance.
    #
    # The evidence was in the output and went unread: the split reported 4,064
    # fraud in the test set, and the scored rows reported 3.915% of 118,108,
    # which is 4,624. Two different counts for the same rows.
    return out.reset_index(drop=True)


def build_sequences(df: pd.DataFrame, codes: dict,
                    max_len: int = MAX_SEQ_LEN) -> dict:
    """One sequence per card, chronological, padded to `max_len`.

    Returns arrays shaped for the model plus `row_index`, which maps every
    (sequence, position) back to its original row - without it the predictions
    cannot be scored against the same test set the GBDT used.
    """
    df = df.sort_values(["uid", "TransactionDT"], kind="stable").reset_index()
    original_index = df["index"].values

    static_cat = apply_codes(df, STATIC_CAT, codes)
    dyn_cat = apply_codes(df, DYNAMIC_CAT, codes)
    dyn_num = df[DYNAMIC_NUM].to_numpy(dtype="float32")
    dyn_num = np.nan_to_num(dyn_num, nan=0.0, posinf=0.0, neginf=0.0)
    labels = df["isFraud"].to_numpy(dtype="int64")

    # Card boundaries, from the sort above.
    uid = df["uid"].values
    starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
    ends = np.r_[starts[1:], len(df)]

    # A card longer than max_len becomes several sequences rather than being
    # truncated - dropping a card's early history would discard exactly the
    # trajectory the model exists to read.
    windows = []
    for s, e in zip(starts, ends):
        for w0 in range(s, e, max_len):
            windows.append((w0, min(w0 + max_len, e)))

    n = len(windows)
    out = {
        "static_cat": np.zeros((n, len(STATIC_CAT)), dtype="int32"),
        "dyn_cat": np.zeros((n, max_len, len(DYNAMIC_CAT)), dtype="int32"),
        "dyn_num": np.zeros((n, max_len, len(DYNAMIC_NUM)), dtype="float32"),
        "sig_cat": np.zeros((n, max_len, 1), dtype="int64"),
        "pad_mask": np.ones((n, max_len), dtype=bool),
        "row_index": np.full((n, max_len), -1, dtype="int64"),
    }
    for i, (w0, w1) in enumerate(windows):
        length = w1 - w0
        out["static_cat"][i] = static_cat[w0]
        out["dyn_cat"][i, :length] = dyn_cat[w0:w1]
        out["dyn_num"][i, :length] = dyn_num[w0:w1]
        out["sig_cat"][i, :length, 0] = labels[w0:w1]
        out["pad_mask"][i, :length] = False
        out["row_index"][i, :length] = original_index[w0:w1]

    return out
