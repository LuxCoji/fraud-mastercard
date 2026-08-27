# Step 11 - Cross-border risk: real, and already handled

Foreign transactions are more fraudulent, so the obvious move was to give them a
lower step-up threshold - the same trick that worked for amount. It was tested
and **rejected**, and the reason is more useful than the change would have been.

## The segments

Measured on the held-out card period, `addr2` being the billing country code and
the home country learned from the training period only:

| Segment | Share | Fraud rate | Risk | Frauds |
| --- | --- | --- | --- | --- |
| domestic | 89.5% | 2.28% | 1.0x | 2,414 |
| cross-border | 0.4% | 8.77% | 3.8x | 37 |
| **country unknown** | **10.1%** | **13.50%** | **5.9x** | **1,613** |

The expected finding was cross-border risk. The actual finding is different:
**cross-border carries higher risk but almost no volume** - 37 frauds - while a
**missing** country code is riskier still, twenty-five times more common, and
accounts for **40% of all fraud in a tenth of traffic**.

Missingness is not a data-quality nuisance on this rail. A transaction arriving
without billing geography is already behaving unusually.

## Adjusting the threshold made things much worse

At an identical alert budget:

| Policy | Value recall | Customers flagged |
| --- | --- | --- |
| Flat | 30.7% | 511 |
| Amount-aware | **44.2%** | 1,079 |
| Amount + segment | **12.6%** | 1,748 |

**-59% on value recall.** The calibration also failed to converge on the target
rate (2.68% against 2.00%), because dividing thresholds by a risk multiple drove
enough of them into the floor that the alert rate could not be brought back
down.

## Why: the model already knows

| Segment | Fraud rate | Mean score | Score ratio |
| --- | --- | --- | --- |
| domestic | 2.283% | 0.0165 | 1.00x |
| cross-border | 8.768% | 0.0918 | 5.58x |
| unknown | 13.504% | 0.1338 | **8.13x** |

The model scores unknown-country transactions **8.13x** higher than domestic
ones, against a **5.91x** difference in actual fraud rate. It has already priced
the segment in - it slightly *over*-prices it - because `addr2` is one of its
input features.

Lowering the threshold on top of that counts the same evidence twice.

## The principle worth keeping

This is the distinction between the two adjustments, and it explains why one
worked and the other did not:

| | What it changes | Can the model see it? |
| --- | --- | --- |
| **Amount** | the **cost** of a mistake | **No** - it optimises a count-based loss |
| **Segment** | the **probability** of fraud | **Yes** - `addr2` is a feature |

**A threshold should only be adjusted for things outside the model's view.**
Anything the features already capture belongs in the score, not in the policy.

That rule generalises: the same test - does the score ratio already track the
risk ratio? - should be run before adding any future segment adjustment, and it
takes about ten lines.

## What was kept

`src/policy/segments.py` remains, as measurement rather than policy. The segment
breakdown is worth reporting on a dashboard - "40% of our fraud arrives without a
country code" is an actionable operational fact even though it is not a
threshold change. `SEGMENT_AWARE_STEP_UP` is off, with the reasoning recorded
next to the flag.
