# Step 6 - Latency: the system was 7.8x over budget, and is now 12x under it

A card authorisation has roughly 100 ms end to end. Nothing had been timed, so
the first measurement was the point.

## Before

| Stage | p50 | p99 |
| --- | --- | --- |
| LightGBM inference (batch 1) | 3.4 ms | 13.2 ms |
| XGBoost inference (batch 1) | 16.7 ms | 106.2 ms |
| Blend (both models) | 30.1 ms | 58.6 ms |
| **Feature building (one transaction)** | **243.0 ms** | **476.4 ms** |
| **END TO END** | **350.1 ms** | **777.3 ms** |

**777 ms against a 100 ms budget.** The models were never the problem - feature
building was **82% of the time**.

## Why

`CardFeatureBuilder.transform` was written for batches of 472,432 rows and is
good at that. Called on one transaction it still pays for DataFrame
construction, thirty `Series.map` calls against lookup tables, and column
alignment - all of it fixed overhead that a single row cannot amortise.

Not slow code. The wrong shape for single-transaction scoring.

## The fix: move work from request time to build time

Nothing about *what* is computed changed. Only *when*:

| Offline, once | Online, per transaction |
| --- | --- |
| per-card aggregates, frequency counts, category codes | dictionary lookups |
| pandas Series | plain dicts |
| - | one preallocated numpy array, written by index |

Of 292 features, 217 are read off the transaction, 71 are dictionary lookups,
and 4 are ratios computed from those lookups. None needs pandas.

## After

| Stage | p50 | p99 |
| --- | --- | --- |
| Feature building | **0.192 ms** | **0.265 ms** |
| Features + model | **1.392 ms** | **3.238 ms** |

**Feature building is 1,266x faster** (243.0 -> 0.192 ms at p50). End to end sits
at **3.2 ms p99 against a 100 ms budget** - about 3% of it.

## Verified identical, not assumed identical

A speed change that quietly alters a feature is worse than a slow one, so
`verify_matches_batch` compares both paths element by element across 100
transactions and 292 features:

**Match rate 100.0000%. Zero mismatches.**

Getting there took three real bugs, each of which would have silently degraded
the model rather than raising an error:

1. **The wrong exclusion set.** Features were skipped by *source column* rather
   than *produced name*, so `card1`, `card2`, `card3` and `P_emaildomain` - which
   are frequency-encoding inputs *and* model features in their own right - came
   back NaN. 150 values wrong.
2. **NaN cannot be a dict key.** `value_counts(dropna=False)` keeps a NaN bucket,
   and for `addr1` that bucket holds 53,761 rows - but `dict.get(nan)` can never
   find it, because `nan != nan`. The NaN count is now stored separately.
3. **Ordering.** The batch path normalises the D columns *before* computing
   `outsider15`, so it compares `D1` against a normalised `D15`. Comparing raw
   values gives a different flag.

All three were found by the verification, not by inspection. None would have
thrown an exception in production.

## Note on the "before" numbers

They were measured on a loaded machine (a browser and editor holding ~5 GB), so
treat them as upper bounds. XGBoost's 106 ms p99 against the blend's 58 ms is
not physically possible and is measurement noise. The direction and the
magnitude of the fix are not in doubt; the exact millisecond figures are soft.
