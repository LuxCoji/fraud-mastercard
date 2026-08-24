# Step 2 Results - Card rail (IEEE-CIS)

Measured on the held-out final 42 days (118,108 transactions, 4,064 fraud,
3.441% base rate). Every encoder fitted on the training period only.

## Headline

| | Logistic (floor) | LightGBM | 
| --- | --- | --- |
| AUC | 0.8160 | **0.9226** |
| PR-AUC | 0.2043 | **0.5691** |
| Brier | 0.03670 | **0.02119** |
| Brier baseline | 0.03323 | 0.03323 |
| Calibrated? | **No - worse than baseline** | Yes |

LightGBM's PR-AUC is **2.79x** the logistic floor, so it earns its latency.
Early stopping chose 1,172 rounds against the published 800 - as expected once
the split changed.

## Against the previous project

The previous attempt scored IEEE-CIS at **0.0730 PR-AUC** (best fused model,
`docs/PHASE3_RESULTS.md`). This run scores **0.5691** on the same source.

That is **7.8x**, and the cause is visible in the feature importances below:
`V258` and `V257` are the two strongest features in the model, and the V block
is exactly what the previous project discarded as "semantically unrecoverable".

## Against the notebook

Deotte's kernel reports 0.9600 AUC. This run reports 0.9226 - **0.037 lower**,
which is the predicted direction and roughly the predicted size. The difference
is the two things we removed:

- UID post-processing, which overwrites a prediction with the average of that
  card's *other* transactions, including later ones.
- Encoders fitted on `concat([train, test])` rather than train alone.

Both are legitimate on Kaggle and neither exists at authorisation time.

## Operating points - LightGBM

| Flag rate | Fraud caught | Precision | Genuine customers hit | False alarms per catch |
| --- | --- | --- | --- | --- |
| 0.1% | 2.9% | 100.0% | 0.00% | 0.0 |
| 0.5% | 13.6% | 93.2% | 0.04% | 0.1 |
| **1.0%** | **26.1%** | **89.7%** | **0.11%** | **0.1** |
| **2.0%** | **42.8%** | **73.6%** | **0.55%** | **0.4** |
| 3.0% | 50.9% | 58.4% | 1.29% | 0.7 |
| 5.0% | 60.2% | 41.4% | 3.03% | 1.4 |
| 10.0% | 74.1% | 25.5% | 7.72% | 2.9 |

The 0.5%-1% band is unusually strong: 90%+ precision means roughly nine of every
ten flags is genuinely fraud. Below 2% the model is confident and cheap; past 3%
precision falls away quickly.

## Top features by gain

| Feature | Origin |
| --- | --- |
| V258, V257, V294, V187 | Raw V block - **discarded by the previous project** |
| C13, C14, C1, C4, C8 | Raw counting columns |
| D2 | Raw time delta |
| TransactionAmt_card1_addr1_P_emaildomain_std | Ours, from the notebooks |
| TransactionAmt_card1_mean | Ours, from the notebooks |
| TransactionAmt_card1_addr1_mean | Ours, from the notebooks |

Four of the top fifteen are the per-card aggregation features, so the notebook's
feature engineering is carrying real weight - but the V columns carry more.

## One honest caveat on the number

Within the training period, a card's aggregate statistics include that card's
own later training transactions. That is a mild self-leak *during fitting*, but
test features are built from training-period statistics only, so it cannot
inflate the reported test score. If anything it creates a small train/test
feature mismatch that costs a little accuracy. The 0.9226 is honest or slightly
conservative, not optimistic.

## Calibration note

The logistic floor is **worse than the base-rate baseline** on Brier (0.03670 vs
0.03323) - it ranks acceptably but its probabilities are not trustworthy.
LightGBM is comfortably better (0.02119). Since every policy threshold is a
probability, this alone would rule the logistic model out of production even if
its ranking were competitive.
