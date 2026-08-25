# Step 2 Results - Card rail (IEEE-CIS)

Measured on the held-out final 42 days (118,108 transactions, 4,064 fraud,
3.441% base rate). Every encoder fitted on the training period only.

## Final numbers

| | Logistic (floor) | LightGBM |
| --- | --- | --- |
| AUC | 0.8202 | **0.9222** |
| PR-AUC | 0.3037 | **0.5843** |
| Brier | 0.02926 | **0.02012** |
| Brier baseline | 0.03323 | 0.03323 |

LightGBM's PR-AUC is 1.92x the floor, so it earns its latency. Early stopping
chose 1,044 rounds against the published 800 - the expected consequence of
changing the split.

## Operating points - LightGBM

| Flag rate | Fraud caught | Precision | Genuine customers hit | False alarms per catch |
| --- | --- | --- | --- | --- |
| 0.1% | 2.8% | 96.6% | 0.00% | 0.0 |
| 0.5% | 13.9% | 95.3% | 0.02% | 0.0 |
| 1.0% | 26.6% | 91.5% | 0.09% | 0.1 |
| **2.0%** | **45.3%** | **78.0%** | **0.46%** | **0.3** |
| 3.0% | 52.2% | 59.9% | 1.25% | 0.7 |
| 5.0% | 60.2% | 41.5% | 3.03% | 1.4 |
| 10.0% | 73.0% | 25.1% | 7.76% | 3.0 |

**Selected operating point: 2%.** Catches 45.3% of fraud while touching 0.46% of
genuine customers - roughly 1 in 217. Changing it later requires no retraining:
it is a threshold applied to scores the model already produces.

## What the three improvements were worth

| Stage | AUC | PR-AUC | Fraud caught at 2% | Genuine hit at 2% |
| --- | --- | --- | --- | --- |
| Baseline features | 0.9226 | 0.5691 | 42.8% | 0.55% |
| + UID, D-normalisation, counts | 0.9250 | 0.5857 | 45.2% | 0.46% |
| + M4 bug fix | 0.9222 | 0.5843 | 45.3% | 0.46% |

The feature work moved the 2% operating point from 42.8% to 45.3% of fraud
caught while *reducing* genuine customers touched from 0.55% to 0.46% - better
on both axes at once, which is the only kind of improvement worth having.

Six of the top fifteen features by gain are the new per-UID aggregates
(`C13_uid_mean`, `TransactionAmt_uid_std`, `TransactionAmt_uid_mean`,
`D4_uid_std`, `D15_uid_mean`, `D4_uid_mean`).

### The M4 fix changed nothing measurable

M1-M3 and M5-M9 are T/F flags, but M4 is an ordered grade with values M0, M1,
M2. The first implementation mapped every M column with the T/F table, which
blanked M4 entirely - a real bug, caught from a scikit-learn imputer warning.

Repairing it took M4 from 0% to 51.9% populated and moved PR-AUC from 0.5857 to
0.5843. That is a **0.24% relative change on a single seed, which is noise**. The
honest conclusion is not that the fix helped or hurt, but that M4 carries no
signal this model can use. The fix is kept because the code is now correct, not
because it bought anything.

## Against the previous project

The previous attempt scored IEEE-CIS at **0.0730 PR-AUC** (best fused model,
`docs/PHASE3_RESULTS.md`). This run scores **0.5843** on the same source - **8.0x**.

The cause is legible in the feature importances: `V258` and `V257` are the two
strongest features in the model, and the V block is exactly what that project
discarded as semantically unrecoverable.

## Against the notebook

Deotte's kernel reports 0.9600 AUC. This run reports 0.9222 - 0.038 lower, the
predicted direction and roughly the predicted size. The gap is the two things we
removed:

- UID post-processing, which overwrites a prediction with the average of that
  card's *other* transactions, including later ones.
- Encoders fitted on `concat([train, test])` rather than train alone.

Both are legitimate on Kaggle and neither exists at authorisation time.

## Caveat on the number

Within the training period, a card's aggregate statistics include that card's
own later training rows. That is a mild self-leak *during fitting*, but test
features are built from training-period statistics only, so it cannot inflate
the reported test score. If anything it creates a small train/test feature
mismatch that costs a little accuracy. The 0.9222 is honest or slightly
conservative, not optimistic.

## Not yet done

- Multi-seed runs. Every number here is a single seed, so differences below
  roughly 1% relative should not be read as real.
- Adding back some of the 219 dropped V columns. V258 and V257 dominate the
  model, so the block matters more than the correlation analysis assumed.
- The three-band policy (allow / step-up / block). Currently a single threshold,
  which forces every flag to be a hard decision.
- Calibration fitting. Brier is already better than baseline, but no explicit
  calibration step has been applied.

---

# Experiment C - Do the 219 discarded V columns matter?

Deotte's correlation analysis kept 120 of the 339 V columns. Since `V258` and
`V257` turned out to be the two strongest features in the model, the block was
worth re-testing.

| | Curated (120 V) | All (339 V) |
| --- | --- | --- |
| Features | 292 | 511 |
| AUC | 0.9222 | 0.9227 |
| PR-AUC | 0.5843 | **0.5914** |
| Brier | 0.02012 | 0.01997 |
| Fraud caught at 2% | 45.3% | 45.8% |
| Genuine hit at 2% | 0.456% | 0.438% |

**PR-AUC change: +1.22%.**

## Verdict: not proven

Before running this, the stated bar was that anything under roughly 1% on a
single seed is noise. The result came in at +1.22% - barely over that line, and
a single seed cannot reliably separate +1.22% from zero.

What is true is that all four measures moved the same direction, which is mildly
suggestive. What is not true is that this is a demonstrated improvement.

The cost is concrete: 511 features against 292, so 75% more to compute at
authorisation time and more memory in the feature store, for a gain that has not
cleared its own noise floor.

**Kept at 120 columns.** Revisit only if a multi-seed run shows the gap holding.

## Settled - three seeds each

| | 120 V columns | 339 V columns |
| --- | --- | --- |
| PR-AUC mean | **0.5865** | 0.5823 |
| PR-AUC spread | +-0.0024 | +-0.0021 |
| AUC mean | 0.9243 | 0.9244 |
| Recall at 2% | **45.19%** | 44.85% |
| Rounds chosen | 990 / 1034 / 1097 | 900 / 870 / 1015 |

Per seed:

| Seed | 120 V | 339 V |
| --- | --- | --- |
| 42 | 0.5841 | 0.5817 |
| 7 | 0.5888 | 0.5806 |
| 2024 | 0.5867 | 0.5846 |

**Mean gap: -0.71%, and 339 loses on all three seeds.**

The single-seed +1.22% was noise, and it pointed the wrong way. Adding the 219
columns back is not merely unproven - it is consistently, mildly *worse*.

A plausible mechanism: `colsample_bytree` is 0.5, so each tree sees half the
feature set. Padding that set with 219 near-duplicates lowers the chance any
given tree draws an informative column. The dropped columns are not neutral
ballast; they crowd out the ones that matter. The shorter runs chosen by early
stopping (870-1015 rounds against 990-1097) point the same way.

Seed spread is +-0.0024, about 0.4% relative. The "1% is noise" bar set earlier
was roughly right in magnitude - the single comparison simply landed on a fluke
large enough to clear it.

**Decision: 120 columns, confirmed.** Deotte's correlation analysis was correct.

---

# Experiment D - Does blending LightGBM and XGBoost help?

Three seeds each. Decision rule fixed before results: adopt the blend only if
its mean PR-AUC beats LightGBM by more than the combined seed spread, and it
wins on every seed.

## Result: REAL

| Model | PR-AUC mean | Spread | Recall at 2% |
| --- | --- | --- | --- |
| **Blend (LGB + XGB)** | **0.5948** | +-0.0033 | **46.67%** |
| LightGBM alone | 0.5865 | +-0.0024 | 45.19% |
| XGBoost alone | 0.5663 | +-0.0170 | 46.10% |

Per seed:

| Seed | LightGBM | XGBoost | Blend |
| --- | --- | --- | --- |
| 42 | 0.5841 | 0.5743 | **0.5982** |
| 7 | 0.5888 | 0.5468 | **0.5916** |
| 2024 | 0.5867 | 0.5778 | **0.5945** |

Gap +1.41%, spread 0.40%, wins all three seeds. **Passes.**

At the 2% operating point the blend catches **46.67%** of fraud against
LightGBM's **45.19%** - 1.48 points, or roughly **60 more frauds caught per
118,000 transactions**.

## XGBoost alone is NOT the answer

Its mean PR-AUC is 3.45% *below* LightGBM and it loses on seeds individually,
so it fails the same rule. It is also strikingly unstable: spread +-0.0170
against LightGBM's +-0.0024, a seven-fold difference, with early stopping
landing anywhere between 1,743 and 2,913 rounds.

That instability is exactly why the blend works. Averaging cancels the noise:
the blend's spread (+-0.0033) is closer to LightGBM's than to XGBoost's, while
its mean beats both. This is an ensemble effect, not one model quietly winning.

## Depth 8 rather than Deotte's depth 12

Depth 12 could not run on this machine - roughly 3-4 GB against ~1.5 GB free,
and it was killed three times at the same point. Depth 8 plus `QuantileDMatrix`
fits.

On seed 42, where both were measured:

| | XGBoost alone | Blend |
| --- | --- | --- |
| depth 8 | 0.5743 | 0.5982 |
| depth 12 | 0.5972 (+4.0%) | 0.6017 (+0.6%) |

Depth 12 makes XGBoost substantially better but the blend barely moves.
LightGBM already covers what the weaker XGBoost misses, so improving XGBoost
adds little on top. **+0.6% is not worth 3x the inference cost**, though that
rests on one seed.

## Outstanding: the blend is not calibrated

Rank-averaging deliberately discards calibration - a rank of 0.5 is not a 50%
chance of fraud. Measured Brier is **0.2985** against LightGBM's 0.0202 and a
base-rate baseline of 0.0332.

This does not affect anything reported here: every operating point is computed
from the ranking, and the 2% cut only asks which transactions sit at the top.
But it does mean the blend **cannot drive a probability threshold until a
calibration layer is fitted on top**. That is required work before this ships,
not an optional refinement.
