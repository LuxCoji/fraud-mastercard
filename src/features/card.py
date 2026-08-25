"""Feature building for the card rail.

The recipe comes from the reference notebooks (see
`reports/step2_notebook_analysis.md`). One thing is done differently, and it is
the whole point: every encoder is fitted on the training rows only and then
applied forward.

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

SECONDS_PER_DAY = 24 * 60 * 60

# V columns that survived Chris Deotte's correlation analysis. The other 219 are
# near-duplicates of these. Source: kaggle.com/cdeotte/eda-for-columns-v-and-id
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

# D columns count days since some past event, so they climb steadily as the
# calendar advances and a model trained on early values meets different numbers
# at test time. Subtracting the transaction day turns each into a fixed
# reference date that does not drift. D1, D2, D3, D5 and D9 are left alone:
# D1 defines the UID, and the notebooks found the others do not benefit.
D_NORMALISE = [4, 6, 7, 8, 10, 11, 12, 13, 14, 15]

FREQUENCY_COLS = [
    "addr1", "card1", "card2", "card3", "P_emaildomain",
    "card1_addr1", "card1_addr1_P_emaildomain", "uid",
]

# Per-UID means and spreads: how does this transaction compare with what this
# card normally does?
AGG_MEAN_STD = ["TransactionAmt", "D4", "D9", "D10", "D15"]
AGG_MEAN_ONLY = [f"C{i}" for i in range(1, 15) if i != 3] + [f"M{i}" for i in range(1, 10)]
AGG_STD_ONLY = ["C14"]

# Per-UID distinct counts: how many different emails, devices or amounts has
# this card touched? A card spraying many distinct values behaves differently
# from one with a settled pattern.
AGG_NUNIQUE = [
    "P_emaildomain", "dist1", "DT_M", "id_02", "cents",
    "C13", "V314", "V127", "V136", "V309", "V307", "V320",
]

# Coarser group keys, kept alongside the UID because they still generalise to
# cards the UID cannot resolve.
COARSE_GROUPS = ["card1", "card1_addr1", "card1_addr1_P_emaildomain"]
COARSE_VALUES = ["TransactionAmt", "D9", "D11"]

# M1-M3 and M5-M9 are T/F flags. M4 is the odd one out: its values are M0, M1
# and M2, an ordered match-quality grade. Mapping it with the T/F table blanks
# the column entirely, so it gets its own mapping.
M_FLAG_COLUMNS = [f"M{i}" for i in (1, 2, 3, 5, 6, 7, 8, 9)]
M_GRADE_MAP = {"M0": 0.0, "M1": 1.0, "M2": 2.0}


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local features. These need no fitting, so they cannot leak."""
    out = df.copy()
    day = out["TransactionDT"] / SECONDS_PER_DAY

    # The decimal part of the amount. Round numbers behave differently from
    # amounts like 59.97, and currency conversion leaves a distinctive tail.
    out["cents"] = (out["TransactionAmt"] - np.floor(out["TransactionAmt"])).astype("float32")

    out["hour"] = ((out["TransactionDT"] / 3600) % 24).astype("float32")
    out["day_of_week"] = (day % 7).astype("float32")
    out["DT_M"] = np.floor(day / 30).astype("float32")

    # Turn the M columns into numbers so that a per-card mean of them means
    # something.
    for col in M_FLAG_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map({"T": 1.0, "F": 0.0}).astype("float32")
    if "M4" in out.columns:
        out["M4"] = out["M4"].map(M_GRADE_MAP).astype("float32")

    # Remove the calendar drift from the D columns.
    for i in D_NORMALISE:
        col = f"D{i}"
        if col in out.columns:
            out[col] = (out[col] - day).astype("float32")

    # Group keys. None is fed to the model - they exist so the encoders below
    # have something card-shaped to group by.
    out["card1_addr1"] = out["card1"].astype(str) + "_" + out["addr1"].astype(str)
    out["card1_addr1_P_emaildomain"] = (
        out["card1_addr1"] + "_" + out["P_emaildomain"].astype(str)
    )

    # The UID: approximately one physical credit card. D1 is days since the card
    # was registered, so day - D1 is the registration date itself, which stays
    # constant across every transaction that card ever makes.
    out["uid"] = out["card1_addr1"] + "_" + np.floor(day - out["D1"]).astype(str)

    # A card whose days-since-first-seen and days-since-last-change disagree by
    # more than three days is behaving oddly.
    out["outsider15"] = (np.abs(out["D1"] - out["D15"]) > 3).astype("float32")

    return out


class CardFeatureBuilder:
    """Fits encoders on training rows, applies them to any rows.

    Everything learned from data lives in this object, so it is impossible to
    fit on test rows by accident: `fit` is called once, with train only.
    """

    def __init__(self, keep_v: list[int] | str = "curated") -> None:
        """`keep_v` selects which V columns reach the model.

        "curated" uses Deotte's 120-column shortlist; "all" keeps all 339, which
        is the experiment testing whether the 219 dropped as redundant were
        carrying anything after all. A list of integers picks them explicitly.
        """
        self.keep_v = keep_v
        self.frequency_maps: dict[str, pd.Series] = {}
        # name -> (group column, lookup table)
        self.aggregate_maps: dict[str, tuple[str, pd.Series]] = {}
        self.category_maps: dict[str, dict] = {}
        self._fitted = False

    def _fit_group_stats(self, df: pd.DataFrame, group_col: str,
                         value_cols: list[str], how: str) -> None:
        for value_col in value_cols:
            if value_col not in df.columns:
                continue
            name = f"{value_col}_{group_col}_{how}"
            self.aggregate_maps[name] = (group_col, df.groupby(group_col)[value_col].agg(how))

    def fit(self, train: pd.DataFrame) -> "CardFeatureBuilder":
        df = _add_derived_columns(train)

        for col in FREQUENCY_COLS:
            self.frequency_maps[col] = df[col].value_counts(dropna=False)

        self._fit_group_stats(df, "uid", AGG_MEAN_STD, "mean")
        self._fit_group_stats(df, "uid", AGG_MEAN_STD, "std")
        self._fit_group_stats(df, "uid", AGG_MEAN_ONLY, "mean")
        self._fit_group_stats(df, "uid", AGG_STD_ONLY, "std")
        self._fit_group_stats(df, "uid", AGG_NUNIQUE, "nunique")

        for group_col in COARSE_GROUPS:
            self._fit_group_stats(df, group_col, COARSE_VALUES, "mean")
            self._fit_group_stats(df, group_col, COARSE_VALUES, "std")

        # Stable integer codes for text columns, learned from train only. A
        # value first seen at test time becomes -1, which is honest: at that
        # moment the system genuinely has never encountered it.
        for col in df.select_dtypes(include=["object"]).columns:
            categories = df[col].astype("category").cat.categories
            self.category_maps[col] = {value: code for code, value in enumerate(categories)}

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call fit() on the training rows first")

        out = _add_derived_columns(df)

        for col, counts in self.frequency_maps.items():
            out[f"{col}_count"] = out[col].map(counts).astype("float32")

        for name, (group_col, lookup) in self.aggregate_maps.items():
            out[name] = out[group_col].map(lookup).astype("float32")

        # How far this amount sits from what this card normally spends.
        for group_col in ["uid"] + COARSE_GROUPS:
            mean_col = f"TransactionAmt_{group_col}_mean"
            std_col = f"TransactionAmt_{group_col}_std"
            if mean_col in out.columns and std_col in out.columns:
                spread = out[std_col].replace(0, np.nan)
                out[f"amt_vs_{group_col}"] = (
                    (out["TransactionAmt"] - out[mean_col]) / spread
                ).astype("float32")

        for col, mapping in self.category_maps.items():
            out[col] = out[col].map(mapping).fillna(-1).astype("int32")

        # The UID and its building blocks are deliberately not model inputs.
        # The aggregations carry the card behaviour; the identifier itself would
        # only invite the model to memorise individual cards.
        drop = ["isFraud", "TransactionID", "TransactionDT",
                "uid", "card1_addr1", "card1_addr1_P_emaildomain"]
        if self.keep_v == "all":
            dropped_v = []
        else:
            selected = KEEP_V if self.keep_v == "curated" else self.keep_v
            keep = {f"V{i}" for i in selected}
            dropped_v = [c for c in out.columns
                         if c.startswith("V") and c[1:].isdigit() and c not in keep]

        out = out.drop(columns=[c for c in drop + dropped_v if c in out.columns])
        return out.select_dtypes(include=["number"])
