"""Feature building for the wire rail (IBM AML).

Laundering is a shape, not a transaction. The 370 labelled rings in
`HI-Small_Patterns.txt` are CYCLE, FAN-IN, FAN-OUT, GATHER-SCATTER,
SCATTER-GATHER, BIPARTITE, STACK and RANDOM - eight topologies, none of which
is visible in a single row. A wire of 5,000 dollars from A to B looks identical
whether it is a salary or the third hop of a layering chain.

So the row is not the unit of evidence here. The *account* is. This module
summarises each account's behaviour across the training period - how many
counterparties it touches, whether money passes straight through it, how
lopsided its in and out flows are - and attaches the sender's and receiver's
summaries to each transaction.

Everything is fitted on the training period only and applied forward, for the
same reason as the card rail: an account's future must not inform its past.

A caveat that belongs in every result from this rail: **IBM AML is synthetic.**
It was checked for the failure that made PaySim unusable - a feature that
separates the classes perfectly - and it is clean: its strongest single signal
(ACH payment format) covers 86.6% of laundering but also 11.8% of legitimate
traffic, which is a correlation rather than a giveaway. Clean is not the same as
real. Numbers from here are plausible, not proven.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENDER = "Account"
RECEIVER = "Account.1"
LABEL = "Is Laundering"


def _account_stats(df: pd.DataFrame, key: str, other: str,
                   prefix: str) -> pd.DataFrame:
    """Summarise what each account does on one side of the ledger.

    `key` is the account being summarised, `other` its counterparty column.
    """
    g = df.groupby(key)
    stats = pd.DataFrame({
        f"{prefix}_txns": g.size(),
        f"{prefix}_partners": g[other].nunique(),
        f"{prefix}_total": g["Amount Paid"].sum(),
        f"{prefix}_mean": g["Amount Paid"].mean(),
        f"{prefix}_max": g["Amount Paid"].max(),
    })
    # A high ratio means many one-off counterparties rather than a settled
    # relationship - the signature of fan-out and scatter patterns.
    stats[f"{prefix}_partner_ratio"] = (
        stats[f"{prefix}_partners"] / stats[f"{prefix}_txns"]
    )
    return stats.astype("float32")


class WireFeatureBuilder:
    """Fits account-level graph statistics on training rows only.

    `with_patterns` adds explicit counts of the laundering shapes themselves -
    tight cycles, scatter-gather widths, fan-out bursts. The IBM paper behind
    this dataset reports that counting the patterns is what lets a gradient
    boosted model beat every GNN they tested, so it is off by default only so
    the two feature sets can be compared honestly.
    """

    def __init__(self, with_patterns: bool = True,
                 with_chains: bool = False,
                 min_account_txns: int = 1,
                 max_cycles: int | None = None,
                 cycle_time_budget_s: float | None = None) -> None:
        self.with_patterns = with_patterns
        self.with_chains = with_chains
        # Bounds for large datasets; inert at their defaults.
        self.min_account_txns = min_account_txns
        self.max_cycles = max_cycles
        self.cycle_time_budget_s = cycle_time_budget_s
        self.out_stats: pd.DataFrame | None = None
        self.in_stats: pd.DataFrame | None = None
        self.pattern_stats: pd.DataFrame | None = None
        self.format_codes: dict[str, int] = {}
        self.currency_codes: dict[str, int] = {}
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> "WireFeatureBuilder":
        self.out_stats = _account_stats(train, SENDER, RECEIVER, "sent")
        self.in_stats = _account_stats(train, RECEIVER, SENDER, "recv")

        if self.with_patterns:
            from features.graph_patterns import build_pattern_features
            frame = train if "_ts" in train.columns else train.assign(
                _ts=pd.to_datetime(train["Timestamp"], format="%Y/%m/%d %H:%M",
                                   errors="coerce"))
            self.pattern_stats = build_pattern_features(
                frame, with_chains=self.with_chains,
                min_account_txns=self.min_account_txns,
                max_cycles=self.max_cycles,
                cycle_time_budget_s=self.cycle_time_budget_s)

        self.format_codes = {v: i for i, v in
                             enumerate(sorted(train["Payment Format"].dropna().unique()))}
        currencies = pd.concat([train["Payment Currency"],
                                train["Receiving Currency"]]).dropna().unique()
        self.currency_codes = {v: i for i, v in enumerate(sorted(currencies))}
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call fit() on the training rows first")

        out = pd.DataFrame(index=df.index)

        # --- the transaction itself -------------------------------------
        amount = df["Amount Paid"].astype("float32")
        out["amount"] = amount
        out["log_amount"] = np.log1p(amount).astype("float32")

        # Money changing size or currency in flight is worth noticing.
        out["amount_gap"] = (df["Amount Paid"] - df["Amount Received"]).abs().astype("float32")
        out["currency_switch"] = (
            df["Payment Currency"] != df["Receiving Currency"]
        ).astype("float32")

        out["payment_format"] = df["Payment Format"].map(self.format_codes).fillna(-1).astype("int16")
        out["currency"] = df["Payment Currency"].map(self.currency_codes).fillna(-1).astype("int16")
        out["same_bank"] = (df["From Bank"] == df["To Bank"]).astype("float32")
        out["self_loop"] = (df[SENDER] == df[RECEIVER]).astype("float32")

        ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
        out["hour"] = ts.dt.hour.astype("float32")
        out["day_of_week"] = ts.dt.dayofweek.astype("float32")

        # --- the two accounts involved ------------------------------------
        sent = self.out_stats.reindex(df[SENDER].values)
        recv = self.in_stats.reindex(df[RECEIVER].values)
        for col in sent.columns:
            out[col] = sent[col].values
        for col in recv.columns:
            out[col] = recv[col].values

        # An account that receives and sends similar totals is a conduit, not a
        # destination. That is what layering looks like from the outside.
        sender_in = self.in_stats.reindex(df[SENDER].values)["recv_total"].values
        sender_out = sent["sent_total"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            out["sender_passthrough"] = np.where(
                sender_in > 0, sender_out / sender_in, np.nan).astype("float32")

        receiver_out = self.out_stats.reindex(df[RECEIVER].values)["sent_total"].values
        receiver_in = recv["recv_total"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            out["receiver_passthrough"] = np.where(
                receiver_in > 0, receiver_out / receiver_in, np.nan).astype("float32")

        # How unusual is this amount for this sender?
        with np.errstate(divide="ignore", invalid="ignore"):
            out["amount_vs_sender_mean"] = (
                amount.values / sent["sent_mean"].values).astype("float32")

        # --- the laundering shapes themselves ------------------------------
        if self.pattern_stats is not None:
            for side, col in (("from", SENDER), ("to", RECEIVER)):
                stats = self.pattern_stats.reindex(df[col].values)
                for name in self.pattern_stats.columns:
                    out[f"{side}_{name}"] = stats[name].values

        return out.replace([np.inf, -np.inf], np.nan)


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return the frame in chronological order.

    IBM AML's rows are not stored in time order - verified in step 1 - so this
    must run before any chronological split or the model trains on its own
    future.
    """
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    return df.assign(_ts=ts).sort_values("_ts", kind="stable").drop(columns="_ts")
