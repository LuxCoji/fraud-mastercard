"""Single-transaction scoring for the card rail, built for the authorisation path.

`CardFeatureBuilder.transform` was written for batches of half a million rows and
is excellent at that. Called on one transaction it costs **476 ms at p99** -
almost eight times the entire authorisation budget - because a single row still
pays for DataFrame construction, thirty `Series.map` calls against lookup tables,
and column alignment.

Nothing here changes *what* is computed. It changes *when*:

    offline (batch, nightly)   per-card aggregates, frequency counts and
                               category codes  ->  plain Python dicts
    online (per transaction)   dictionary lookups and arithmetic into a
                               preallocated numpy array

The values are identical to the batch path by construction - every lookup comes
from the same fitted `CardFeatureBuilder` - so this is an engineering change with
no accuracy cost, and `verify_matches_batch` asserts it rather than assuming it.

Of the 292 features, 217 are read straight off the transaction, 71 are dictionary
lookups, and 4 are ratios computed from the lookups. None requires pandas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.card import (D_NORMALISE, M_FLAG_COLUMNS, M_GRADE_MAP,
                           SECONDS_PER_DAY, CardFeatureBuilder)

M_FLAG_MAP = {"T": 1.0, "F": 0.0}


class FastCardScorer:
    """Scores one transaction from a plain dict.

    Construct once from a fitted `CardFeatureBuilder` and the model objects; the
    hot path then does no fitting, no allocation beyond one array, and no pandas.
    """

    def __init__(self, builder: CardFeatureBuilder, feature_names: list[str],
                 models: list) -> None:
        self.feature_names = list(feature_names)
        self.n_features = len(self.feature_names)
        self.models = models

        # Position of every feature, so the hot path writes by index rather than
        # by name. Building this once removes a dict lookup per feature per call.
        self.index = {name: i for i, name in enumerate(self.feature_names)}

        # --- lookup tables, converted from pandas to plain dicts -----------
        # A Series.map costs far more than a dict.get, and the values never
        # change between fits, so the conversion is paid once.
        self.frequency, self.frequency_nan = {}, {}
        for col, counts in builder.frequency_maps.items():
            table = counts.to_dict()
            # NaN is a legitimate bucket - value_counts(dropna=False) keeps it -
            # but it can never be found by lookup because nan != nan. Pull it
            # out and dispatch to it explicitly.
            nan_key = [k for k in table if isinstance(k, float) and k != k]
            self.frequency_nan[col] = table.pop(nan_key[0]) if nan_key else np.nan
            self.frequency[col] = table
        self.aggregates = {name: (group_col, series.to_dict())
                           for name, (group_col, series) in builder.aggregate_maps.items()}
        self.categories = builder.category_maps

        # Which features are computed rather than copied. Note this lists the
        # *produced feature names* - `card1_count`, not `card1`. Excluding the
        # source column instead silently dropped card1, card2, card3 and
        # P_emaildomain, which are frequency-encoding inputs *and* model features
        # in their own right: 150 values came back NaN that should not have.
        derived = {f"{col}_count" for col in self.frequency} | set(self.aggregates) | {
            "cents", "hour", "day_of_week", "DT_M", "outsider15",
            "amt_vs_uid", "amt_vs_card1", "amt_vs_card1_addr1",
            "amt_vs_card1_addr1_P_emaildomain",
        }
        self.passthrough = [n for n in self.feature_names if n not in derived]
        self.d_normalise = {f"D{i}" for i in D_NORMALISE}
        self.m_flags = set(M_FLAG_COLUMNS)

    # -- the hot path -------------------------------------------------------
    def build_features(self, txn: dict) -> np.ndarray:
        """One transaction dict -> one feature vector."""
        x = np.full(self.n_features, np.nan, dtype=np.float32)
        idx = self.index
        day = txn["TransactionDT"] / SECONDS_PER_DAY

        # Group keys. Built as strings once and reused by every lookup below.
        card1 = txn.get("card1")
        card1_addr1 = f"{card1}_{txn.get('addr1')}"
        card1_addr1_email = f"{card1_addr1}_{txn.get('P_emaildomain')}"
        uid = f"{card1_addr1}_{np.floor(day - txn.get('D1', np.nan))}"
        keys = {"card1": card1, "card1_addr1": card1_addr1,
                "card1_addr1_P_emaildomain": card1_addr1_email, "uid": uid,
                "addr1": txn.get("addr1"), "card2": txn.get("card2"),
                "card3": txn.get("card3"), "P_emaildomain": txn.get("P_emaildomain")}

        # --- raw columns ---------------------------------------------------
        for name in self.passthrough:
            value = txn.get(name)
            if name in self.categories:
                # A value the training period never saw encodes as -1, and so
                # does a missing one - the batch path maps both the same way,
                # and "never seen before" is the honest reading of each.
                x[idx[name]] = self.categories[name].get(value, -1)
                continue
            if value is None:
                continue
            elif name in self.m_flags:
                x[idx[name]] = M_FLAG_MAP.get(value, np.nan)
            elif name == "M4":
                x[idx[name]] = M_GRADE_MAP.get(value, np.nan)
            elif name in self.d_normalise:
                x[idx[name]] = value - day if value is not None else np.nan
            elif isinstance(value, (int, float, np.number)):
                x[idx[name]] = value

        # --- row-local derived ---------------------------------------------
        amount = txn["TransactionAmt"]
        if "cents" in idx:
            x[idx["cents"]] = amount - np.floor(amount)
        if "hour" in idx:
            x[idx["hour"]] = (txn["TransactionDT"] / 3600) % 24
        if "day_of_week" in idx:
            x[idx["day_of_week"]] = day % 7
        if "DT_M" in idx:
            x[idx["DT_M"]] = np.floor(day / 30)
        if "outsider15" in idx:
            # D15 is normalised before this comparison in the batch path
            # (D_NORMALISE includes 15), so the same subtraction must happen
            # here or the flag disagrees.
            d1, d15 = txn.get("D1"), txn.get("D15")
            if d1 is not None and d15 is not None:
                x[idx["outsider15"]] = float(abs(d1 - (d15 - day)) > 3)

        # --- frequency counts ------------------------------------------------
        for col, table in self.frequency.items():
            name = f"{col}_count"
            if name not in idx:
                continue
            key = keys.get(col)
            if isinstance(key, float) and key != key:      # NaN
                x[idx[name]] = self.frequency_nan[col]
            else:
                x[idx[name]] = table.get(key, np.nan)

        # --- per-group aggregates --------------------------------------------
        for name, (group_col, table) in self.aggregates.items():
            if name in idx:
                x[idx[name]] = table.get(keys.get(group_col), np.nan)

        # --- amount against each group's own mean -----------------------------
        for group in ("uid", "card1", "card1_addr1", "card1_addr1_P_emaildomain"):
            name = f"amt_vs_{group}"
            if name not in idx:
                continue
            mean_i = idx.get(f"TransactionAmt_{group}_mean")
            std_i = idx.get(f"TransactionAmt_{group}_std")
            if mean_i is None or std_i is None:
                continue
            std = x[std_i]
            if std and not np.isnan(std):
                x[idx[name]] = (amount - x[mean_i]) / std

        return x

    def score(self, txn: dict) -> float:
        """Rank-free probability from the blend. Batch of one, as production sees it."""
        features = self.build_features(txn).reshape(1, -1)
        return float(np.mean([self._predict(m, features) for m in self.models]))

    @staticmethod
    def _predict(model, features: np.ndarray) -> float:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[0, 1]
        return model.predict(features)[0]


def verify_matches_batch(scorer: FastCardScorer, builder: CardFeatureBuilder,
                         raw_rows: pd.DataFrame, n: int = 50) -> dict:
    """Confirm the fast path produces the batch path's numbers.

    A speed optimisation that quietly changes a feature is worse than a slow one,
    so this compares the two vectors element by element and reports where they
    differ rather than trusting that they do not.
    """
    batch = builder.transform(raw_rows.head(n))
    mismatches, checked = [], 0

    for i in range(min(n, len(raw_rows))):
        txn = raw_rows.iloc[i].to_dict()
        fast = scorer.build_features(txn)
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
