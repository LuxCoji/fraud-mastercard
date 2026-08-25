"""Feature building for the transfer rail (PaySim).

PaySim gives you five real columns, so nothing here comes from having clever
inputs - it comes from balance arithmetic. A transfer has a sender balance
before and after, a receiver balance before and after, and an amount. Those five
numbers should reconcile. When they do not, something is wrong.

A representative fraud row:

    TRANSFER  amount 181.00
      oldbalanceOrg  181.00 -> newbalanceOrig    0.00   drained to exactly zero
      oldbalanceDest   0.00 -> newbalanceDest    0.00   receiver never credited

The account is emptied to the cent, and the destination's balance does not move
despite receiving the money. Both are computable in one line each, and together
they carry most of the signal on this rail.

`isFlaggedFraud` is never used. It is the simulator's own rule-based guess, not
an observation, and feeding it to a model would be leaking a partial answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fraud occurs only in these two types. Verified on the full 6,362,620 rows:
# PAYMENT, CASH_IN and DEBIT contain zero fraud between them - 3,592,211 rows,
# 56.5% of all traffic. The router drops them before scoring rather than asking
# a model to learn a fact we already know for certain.
SCOREABLE_TYPES = ("TRANSFER", "CASH_OUT")

# Never a feature: nameOrig/nameDest are identifiers, isFlaggedFraud is a leak.
DROP_COLUMNS = ["nameOrig", "nameDest", "isFraud", "isFlaggedFraud", "step"]

HOURS_PER_DAY = 24


def route(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the payment types that can carry fraud.

    This is the cheapest possible false-positive control: a transaction that is
    provably never fraudulent should never reach a model that might flag it.
    """
    return df[df["type"].isin(SCOREABLE_TYPES)].copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local features only - nothing fitted, so nothing can leak.

    Every column below is computable from the single transaction in front of
    you, which also means the whole feature set is available at authorisation
    time with no feature store lookup.
    """
    out = df.copy()

    # --- the balance reconciliation, the core of this rail ------------------
    # Money left the sender: did the sender's balance drop by the amount?
    out["error_balance_orig"] = (
        out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    ).astype("float32")

    # Money reached the receiver: did the receiver's balance rise by it?
    out["error_balance_dest"] = (
        out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    ).astype("float32")

    # Whether the books balance at all matters separately from by how much.
    out["orig_reconciles"] = (out["error_balance_orig"].abs() < 0.01).astype("float32")
    out["dest_reconciles"] = (out["error_balance_dest"].abs() < 0.01).astype("float32")

    # --- draining behaviour -------------------------------------------------
    # Fraud empties an account; ordinary spending rarely lands on exactly zero.
    out["orig_emptied"] = (
        (out["newbalanceOrig"] == 0) & (out["oldbalanceOrg"] > 0)
    ).astype("float32")

    # How much of what was there did this transaction take?
    out["amount_over_balance"] = (
        out["amount"] / out["oldbalanceOrg"].replace(0, np.nan)
    ).astype("float32")
    out["takes_everything"] = (
        (out["amount"] - out["oldbalanceOrg"]).abs() < 0.01
    ).astype("float32")

    # --- the receiving side -------------------------------------------------
    out["dest_was_empty"] = (out["oldbalanceDest"] == 0).astype("float32")
    out["dest_still_empty"] = (out["newbalanceDest"] == 0).astype("float32")
    # Merchant accounts are prefixed M; they behave differently from consumers.
    out["dest_is_merchant"] = out["nameDest"].str.startswith("M").astype("float32")

    # --- amount and timing --------------------------------------------------
    # Amounts are heavy-tailed; the log is what a tree can actually split on
    # without spending its first few splits undoing the skew.
    out["log_amount"] = np.log1p(out["amount"]).astype("float32")
    out["hour_of_day"] = (out["step"] % HOURS_PER_DAY).astype("float32")
    out["day"] = (out["step"] // HOURS_PER_DAY).astype("float32")

    # --- type ---------------------------------------------------------------
    # Two values after routing, so a single flag rather than one-hot.
    out["is_transfer"] = (out["type"] == "TRANSFER").astype("float32")

    out = out.drop(columns=[c for c in DROP_COLUMNS + ["type"] if c in out.columns])
    return out.select_dtypes(include=["number"])
