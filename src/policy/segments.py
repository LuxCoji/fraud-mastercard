"""Per-segment thresholds: some transactions are riskier before you score them.

Amount already shifts the step-up threshold (Elkan's rule). Geography is the
same argument from the other side: it changes the *prior* probability of fraud
rather than the cost of being wrong.

Measured on the held-out card period, splitting by `addr2` - the billing country
code - with the home country learned from the training period only:

| segment | share of traffic | fraud rate | risk multiple | frauds |
| --- | --- | --- | --- | --- |
| domestic | 89.5% | 2.28% | 1.0x | 2,414 |
| cross-border | 0.4% | 8.77% | 3.8x | 37 |
| **country unknown** | **10.1%** | **13.50%** | **5.9x** | **1,613** |

The headline finding is not the one that was expected. Cross-border carries
higher risk but almost no volume - 37 frauds. **A missing country code is both
riskier and twenty-five times more common**, and it accounts for 40% of all
fraud in the test period while being a tenth of traffic.

That is worth stating plainly: missingness is not a data-quality nuisance here,
it is the signal. A transaction that arrives without billing geography is
already behaving unusually.

The thresholds below divide by the risk multiple, so a segment that is 5.9x
riskier needs proportionally less confidence to be challenged. As with amount,
this applies to **step-up only** - blocking keeps a fixed high bar, because a
wrongly blocked customer costs the same whichever segment they are in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DOMESTIC = "domestic"
CROSS_BORDER = "cross_border"
UNKNOWN = "unknown"

# How far a segment's threshold may be moved. Without a bound, a segment that
# happened to contain few frauds in the training period could have its threshold
# driven to nonsense by a noisy estimate.
MIN_MULTIPLE, MAX_MULTIPLE = 0.5, 6.0

# Below this many frauds a segment's rate is too noisy to act on, and it falls
# back to the baseline. Cross-border has 37 in the test period - enough to
# measure, not enough to be confident about, and this is where that judgement
# lives rather than being buried in a magic number.
MIN_FRAUDS_TO_ADJUST = 30


def segment_of(addr2, home_country) -> str:
    """Which geography segment a transaction belongs to."""
    if addr2 is None or (isinstance(addr2, float) and addr2 != addr2):
        return UNKNOWN
    return DOMESTIC if addr2 == home_country else CROSS_BORDER


def segments_for(addr2: pd.Series, home_country) -> pd.Series:
    """Vectorised form, for scoring a whole period at once."""
    values = addr2.to_numpy()
    missing = pd.isna(values)
    return pd.Series(np.where(missing, UNKNOWN,
                              np.where(values == home_country, DOMESTIC, CROSS_BORDER)),
                     index=addr2.index)


def fit_segment_risk(addr2: pd.Series, labels: np.ndarray,
                     home_country) -> dict[str, float]:
    """Risk multiple per segment, measured against the domestic baseline.

    Fitted on the training period only - a multiple derived from the data the
    model is scored on would be circular.
    """
    seg = segments_for(addr2, home_country)
    rates = {}
    for name in (DOMESTIC, CROSS_BORDER, UNKNOWN):
        mask = (seg == name).to_numpy()
        rates[name] = float(labels[mask].mean()) if mask.any() else np.nan

    baseline = rates.get(DOMESTIC) or np.nan
    multiples = {}
    for name, rate in rates.items():
        frauds = int(labels[(seg == name).to_numpy()].sum())
        if not np.isfinite(rate) or not np.isfinite(baseline) or baseline <= 0 \
                or frauds < MIN_FRAUDS_TO_ADJUST:
            multiples[name] = 1.0
        else:
            multiples[name] = float(np.clip(rate / baseline, MIN_MULTIPLE, MAX_MULTIPLE))
    return multiples


def learn_home_country(addr2: pd.Series):
    """The most common billing country in the training period.

    Not hardcoded: the same code should work for an issuer in any country, and
    hardcoding 87.0 would silently break the moment the data changed.
    """
    mode = addr2.mode()
    return mode.iloc[0] if len(mode) else None


def describe(addr2: pd.Series, labels: np.ndarray, home_country) -> pd.DataFrame:
    """The table above, for whatever period is passed in."""
    seg = segments_for(addr2, home_country)
    multiples = fit_segment_risk(addr2, labels, home_country)

    rows = []
    for name in (DOMESTIC, CROSS_BORDER, UNKNOWN):
        mask = (seg == name).to_numpy()
        if not mask.any():
            continue
        rows.append({"segment": name, "share": float(mask.mean()),
                     "fraud_rate": float(labels[mask].mean()),
                     "frauds": int(labels[mask].sum()),
                     "risk_multiple": multiples[name],
                     "adjusted": multiples[name] != 1.0})
    return pd.DataFrame(rows)
