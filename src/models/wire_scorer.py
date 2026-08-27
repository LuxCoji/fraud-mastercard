"""Single-transaction scoring for the wire rail.

The card rail needed this for latency - 476 ms of pandas overhead per
transaction inside a 100 ms budget. The wire rail runs asynchronously and has no
such constraint, so the reason here is different: the router could not call the
wire model at all. `WireFeatureBuilder.transform` takes a DataFrame, and the
router hands it a dict.

The account-graph features come from a table computed offline over the training
period - degrees, pass-through ratios, cycle and fan counts. Looking them up is
the same operation the card rail performs, and for the same reason: the
expensive part is building the table, not reading it.

An account absent from that table gets NaN rather than a substituted zero. A
zero would claim "this account has no cycles", which is a statement about the
account; NaN says "this account was not in the training period", which is a
statement about our knowledge. LightGBM and XGBoost both handle NaN natively and
learn what missingness means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.wire import LABEL, SENDER, RECEIVER, WireFeatureBuilder


class FastWireScorer:
    """Scores one wire from a plain dict."""

    def __init__(self, builder: WireFeatureBuilder, feature_names: list[str],
                 models: list) -> None:
        self.feature_names = list(feature_names)
        self.n_features = len(self.feature_names)
        self.index = {name: i for i, name in enumerate(self.feature_names)}
        self.models = models

        # pandas Series -> plain dicts, once.
        self.sent = {c: builder.out_stats[c].to_dict() for c in builder.out_stats.columns}
        self.recv = {c: builder.in_stats[c].to_dict() for c in builder.in_stats.columns}
        self.patterns = ({c: builder.pattern_stats[c].to_dict()
                          for c in builder.pattern_stats.columns}
                         if builder.pattern_stats is not None else {})
        self.format_codes = builder.format_codes
        self.currency_codes = builder.currency_codes

    def build_features(self, txn: dict) -> np.ndarray:
        x = np.full(self.n_features, np.nan, dtype=np.float32)
        idx = self.index
        sender, receiver = txn.get(SENDER), txn.get(RECEIVER)

        def put(name: str, value) -> None:
            if name in idx and value is not None:
                x[idx[name]] = value

        amount = float(txn.get("Amount Paid", np.nan))
        received = float(txn.get("Amount Received", amount))
        put("amount", amount)
        put("log_amount", np.log1p(amount))
        put("amount_gap", abs(amount - received))
        put("currency_switch",
            float(txn.get("Payment Currency") != txn.get("Receiving Currency")))
        put("payment_format", self.format_codes.get(txn.get("Payment Format"), -1))
        put("currency", self.currency_codes.get(txn.get("Payment Currency"), -1))
        put("same_bank", float(txn.get("From Bank") == txn.get("To Bank")))
        put("self_loop", float(sender == receiver))

        timestamp = pd.to_datetime(txn.get("Timestamp"), format="%Y/%m/%d %H:%M",
                                   errors="coerce")
        if pd.notna(timestamp):
            put("hour", timestamp.hour)
            put("day_of_week", timestamp.dayofweek)

        # --- the two accounts -------------------------------------------------
        for column, table in self.sent.items():
            put(column, table.get(sender))
        for column, table in self.recv.items():
            put(column, table.get(receiver))

        # An account that receives and sends similar totals is a conduit rather
        # than a destination - what layering looks like from outside.
        sender_in = self.recv.get("recv_total", {}).get(sender)
        sender_out = self.sent.get("sent_total", {}).get(sender)
        if sender_in:
            put("sender_passthrough", sender_out / sender_in if sender_out else np.nan)

        receiver_in = self.recv.get("recv_total", {}).get(receiver)
        receiver_out = self.sent.get("sent_total", {}).get(receiver)
        if receiver_in:
            put("receiver_passthrough", receiver_out / receiver_in if receiver_out else np.nan)

        sender_mean = self.sent.get("sent_mean", {}).get(sender)
        if sender_mean:
            put("amount_vs_sender_mean", amount / sender_mean)

        # --- laundering shapes, attached to both sides -------------------------
        for side, account in (("from", sender), ("to", receiver)):
            for name, table in self.patterns.items():
                put(f"{side}_{name}", table.get(account))

        return x

    def score(self, txn: dict) -> float:
        features = self.build_features(txn).reshape(1, -1)
        return float(np.mean([self._predict(m, features) for m in self.models]))

    @staticmethod
    def _predict(model, features: np.ndarray) -> float:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[0, 1]
        return model.predict(features)[0]


def verify_matches_batch(scorer: FastWireScorer, builder: WireFeatureBuilder,
                         raw_rows: pd.DataFrame, n: int = 50) -> dict:
    """Confirm the fast path reproduces the batch path, feature by feature.

    The card rail's equivalent check found three bugs that would each have
    degraded the model silently, so the wire rail gets the same treatment.
    """
    batch = builder.transform(raw_rows.head(n))
    mismatches, checked = [], 0

    for i in range(min(n, len(raw_rows))):
        fast = scorer.build_features(raw_rows.iloc[i].to_dict())
        slow = batch.iloc[i].values.astype(np.float32)
        both_nan = np.isnan(fast) & np.isnan(slow)
        close = np.isclose(fast, slow, rtol=1e-4, atol=1e-6, equal_nan=True)
        bad = ~(close | both_nan)
        checked += 1
        if bad.any():
            for j in np.flatnonzero(bad)[:5]:
                mismatches.append((scorer.feature_names[j], float(slow[j]), float(fast[j])))

    return {"rows_checked": checked, "mismatches": mismatches,
            "match_rate": 1 - len(mismatches) / max(checked * scorer.n_features, 1)}
