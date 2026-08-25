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
