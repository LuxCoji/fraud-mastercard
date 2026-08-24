"""Feature building for the card rail.

The features come from the reference notebooks (see
`reports/step2_notebook_analysis.md`). One thing is done differently, and it is
the whole point: every encoder is fitted on the training rows only and then
applied forward to the test rows.

The notebooks fit theirs on `pd.concat([train, test])`. That is legitimate on
Kaggle, where the test set is handed to you in full, but a live authorisation
has no access to future transactions. Fitting train-only is what makes the
resulting score honest.

Usage:
    builder = CardFeatureBuilder()
    builder.fit(df.iloc[train_idx])
    X_train = builder.transform(df.iloc[train_idx])
    X_test  = builder.transform(df.iloc[test_idx])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# V columns that survived Chris Deotte's correlation analysis. The other 219 are
# near-duplicates of these; dropping them costs no accuracy and saves memory.
# Source: kaggle.com/cdeotte/eda-for-columns-v-and-id
KEEP_V = [
    1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47,
    48, 54, 56, 59, 62, 65, 67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91, 107,
    108, 111, 115, 117, 120, 121, 123, 124, 127, 129, 130, 136, 138, 139, 142,
    147, 156, 160, 162, 165, 166, 169, 171, 173, 175, 176, 178, 180, 182, 185,
    187, 188, 198, 203, 205, 207, 209, 210, 215, 218, 220, 221, 223, 224, 226,
    228, 229, 234, 235, 238, 240, 250, 252, 253, 257, 258, 260, 261, 264, 266,
    267, 271, 274, 277, 281, 283, 284, 285, 286, 289, 291, 294, 296, 297, 301,
    303, 305, 307, 309, 310, 314, 320,
]

# Counted, not modelled: how often a value appears is itself predictive.
FREQUENCY_COLS = [
    "addr1", "card1", "card2", "card3", "P_emaildomain",
    "card1_addr1", "card1_addr1_P_emaildomain",
]

# Per-card statistics. "How does this amount compare to what this card usually
# spends" is the signal; the group keys stand in for a physical card.
AGG_VALUE_COLS = ["TransactionAmt", "D9", "D11"]
AGG_GROUP_COLS = ["card1", "card1_addr1", "card1_addr1_P_emaildomain"]

SECONDS_PER_DAY = 24 * 60 * 60


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local features. These need no fitting, so they cannot leak."""
    out = df.copy()

    # The decimal part of the amount. Round numbers behave differently from
    # amounts like 59.97, and currency conversion leaves a distinctive tail.
    out["cents"] = (out["TransactionAmt"] - np.floor(out["TransactionAmt"])).astype("float32")

    # Clock position within the day and week. Fraud is not uniform over time.
    day = out["TransactionDT"] / SECONDS_PER_DAY
    out["hour"] = ((out["TransactionDT"] / 3600) % 24).astype("float32")
    out["day_of_week"] = (day % 7).astype("float32")

    # Combination keys. Neither is used as a model input directly - they exist
    # so the frequency and aggregation encoders have something card-shaped to
    # group by.
    out["card1_addr1"] = (
        out["card1"].astype(str) + "_" + out["addr1"].astype(str)
    )
    out["card1_addr1_P_emaildomain"] = (
        out["card1_addr1"] + "_" + out["P_emaildomain"].astype(str)
    )
    return out


class CardFeatureBuilder:
    """Fits encoders on training rows, applies them to any rows.

    Everything learned from data lives in this object, so it is impossible to
    accidentally fit on test rows: `fit` is called once, with train only.
    """

    def __init__(self) -> None:
        self.frequency_maps: dict[str, pd.Series] = {}
        # name -> (group column, lookup table)
        self.aggregate_maps: dict[str, tuple[str, pd.Series]] = {}
        self.category_maps: dict[str, dict] = {}
        self.feature_names: list[str] = []
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> "CardFeatureBuilder":
        df = _add_derived_columns(train)

        # How often each value appears in the training period.
        for col in FREQUENCY_COLS:
            self.frequency_maps[col] = df[col].value_counts(dropna=False)

        # Mean and spread of each value column, per card-like group.
        for value_col in AGG_VALUE_COLS:
            for group_col in AGG_GROUP_COLS:
                grouped = df.groupby(group_col)[value_col]
                base = f"{value_col}_{group_col}"
                self.aggregate_maps[f"{base}_mean"] = (group_col, grouped.mean())
                self.aggregate_maps[f"{base}_std"] = (group_col, grouped.std())

        # Stable integer codes for text columns, learned from train only.
        # Values first seen in test become -1, which is honest: at that moment
        # the system genuinely has never encountered them.
        for col in df.select_dtypes(include=["object"]).columns:
            categories = df[col].astype("category").cat.categories
            self.category_maps[col] = {value: code for code, value in enumerate(categories)}

        self._fitted = True
        self.feature_names = list(self.transform(train.head(100)).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call fit() on the training rows first")

        out = _add_derived_columns(df)

        for col, counts in self.frequency_maps.items():
            out[f"{col}_count"] = out[col].map(counts).astype("float32")

        for name, (group_col, lookup) in self.aggregate_maps.items():
            out[name] = out[group_col].map(lookup).astype("float32")

        # How far this amount sits from what the card normally spends.
        for group_col in AGG_GROUP_COLS:
            mean = out[f"TransactionAmt_{group_col}_mean"]
            std = out[f"TransactionAmt_{group_col}_std"].replace(0, np.nan)
            out[f"amt_vs_{group_col}"] = (
                (out["TransactionAmt"] - mean) / std
            ).astype("float32")

        for col, mapping in self.category_maps.items():
            out[col] = out[col].map(mapping).fillna(-1).astype("int32")

        drop = ["isFraud", "TransactionID", "TransactionDT"]
        keep_v = {f"V{i}" for i in KEEP_V}
        dropped_v = [c for c in out.columns
                     if c.startswith("V") and c[1:].isdigit() and c not in keep_v]

        out = out.drop(columns=[c for c in drop + dropped_v if c in out.columns])
        return out.select_dtypes(include=["number"])
