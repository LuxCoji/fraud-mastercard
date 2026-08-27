# Step 10 - Value-weighted metrics, and the large-fraud blind spot

Counting frauds treats a 2-rupee loss and a 3,000-rupee loss as one event each.
Measuring by value instead surfaced a problem that the headline numbers had been
hiding.

## The model is worst at the frauds that cost most

| | |
| --- | --- |
| Fraud caught **by count** | 45.6% |
| Fraud caught **by value** | **30.7%** |

Detection falls monotonically with amount:

| Fraud amount decile | Caught |
| --- | --- |
| smallest (2-16) | 69.9% |
| 54-66 | 44.6% |
| 98-117 | 29.9% |
| largest (335-3,133) | **24.3%** |

**Of the 20 largest frauds, 1 was caught.** Median caught fraud: 50. Median
missed fraud: 95.

Nothing was done wrong. PR-AUC - the metric everything was optimised against -
counts a 2-rupee fraud and a 3,000-rupee fraud equally, so the model correctly
learned to catch the many small ones.

## Value-weighted *training* barely helps

| | Baseline | weight = amount | weight = sqrt(amount) |
| --- | --- | --- | --- |
| PR-AUC | 0.5170 | 0.4613 | **0.5308** |
| Value recall | 28.4% | 28.3% | 29.7% |
| Genuine value disturbed | 72,746 | **162,784** | 72,372 |
| Top-20 largest caught | 3 | 3 | 3 |

Weighting by raw amount **failed outright**: it stopped 0.2% *less* fraud value
while disturbing **124% more** genuine value - the predicted failure mode, where
a model keener on large amounts also makes larger mistakes. The square-root
version passed, marginally: +4.5% value for -0.5% cost.

Neither fixed the real problem. Still 17 of the 20 largest frauds missed.

## The features were never the problem

Among large transactions alone, the best single feature separates fraud at
**AUC 0.731**, against 0.747 across all transactions. Large frauds are visible.
They were falling below a threshold set for the average transaction.

That reframes the fix entirely: this is a **decision** problem, not a modelling
one, and needs no retraining.

## Elkan's rule

Elkan (2001) gives the cost-minimising boundary directly:

```
threshold* = cost(false positive) / (cost(false positive) + cost(false negative))
```

For fraud, `cost(false negative)` is the amount at risk and varies per
transaction, while `cost(false positive)` is roughly fixed. So the threshold
should not be constant:

```
threshold(amount) = c_fp / (c_fp + amount)
```

A 10,000-rupee transaction needs less confidence to be worth stopping than a
50-rupee one, because being wrong costs the same while being right is worth two
hundred times more.

## Measured, at an identical 2% alert budget

| Policy | Value recall | Count recall | Customers flagged | Top-20 caught |
| --- | --- | --- | --- | --- |
| Fixed threshold | 30.7% | 45.6% | 511 | 1 |
| `amount^0.25` | 39.4% | 41.3% | 689 (1.3x) | 4 |
| **`amount^0.5`** | **44.2%** | 31.7% | **1,078 (2.1x)** | **9** |
| `amount^0.75` | 45.0% | 24.3% | 1,379 (2.7x) | 9 |
| `amount^1.0` (pure Elkan) | 45.4% | 20.2% | 1,552 (3.0x) | 10 |

Every row spends the same 2% of traffic. What changes is *which* transactions
it spends it on.

## The trade-off is real and it is a business decision

Pure Elkan stops **48% more fraud value** and catches **10 of the 20 largest
frauds instead of 1**. It also flags **three times as many genuine customers**,
and count recall falls from 45.6% to 20.2% - it is deliberately letting small
frauds through to spend the budget on large ones.

**`amount^0.5` is the knee of the curve.** It captures 44% of the value gain
(30.7% -> 44.2%) and 9 of the top 20, for 2.1x the customer friction rather than
3x. Past 0.5 the value gain flattens while the cost keeps climbing.

## What this cannot decide on its own

Whether 2.1x the friction is worth 44% more fraud value stopped depends on what
a false positive actually costs, and that is not a property of the model:

- On the **step-up** band it is an OTP - mild, and probably worth it.
- On the **block** band it is a lost sale and an angry customer - much less so.

The sensible shape is to apply the amount-dependent threshold to the step-up
band, where friction is cheap, and keep the block band on a fixed high bar. That
is a policy change and belongs with whoever owns the customer relationship.

**Nothing has been adopted.** The metrics and the threshold rule are implemented
and measured; the choice is recorded as open.
