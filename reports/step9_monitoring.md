# Step 9 - Drift monitoring and model rollback

A fraud model learns the patterns of the period it trained on. Criminals change
tactics and the model degrades quietly: nothing errors, the API keeps returning
scores, and the only symptom is that fewer frauds get caught.

## The timing problem

Ground truth arrives weeks to months late - chargebacks, disputes,
investigations. Waiting for labels to confirm the model broke means finding out
a quarter after it happened. So the monitor watches what is available *now*.

| Signal | Available | Catches |
| --- | --- | --- |
| Alert rate | immediately | capacity problems, threshold drift |
| Score PSI | immediately | the model's confidence shifting |
| Feature PSI | immediately | inputs no longer looking like training |
| Estimated precision | immediately | degradation, *while calibration holds* |
| `label_backfill` | when labels arrive | whether the estimate was honest |

## Estimated performance without labels

For a calibrated model a score **is** a probability, so the expected true
positives over the flagged set is just the sum of their scores. This is the
approach NannyML calls Confidence-Based Performance Estimation, and it works
here because the card rail is Platt-calibrated.

Its limitation is real and worth stating plainly: **it cannot see concept
drift.** If the feature-to-fraud relationship changes, the model becomes
confidently wrong and the estimate goes wrong with it. That is why
`label_backfill` exists - when labels do turn up, the estimate is checked
against them, and a large gap is a more serious finding than the drift it
failed to detect.

Measured on our own held-out data, the estimate was **not** trustworthy:
estimated precision 0.456 against an actual 0.742. The model is materially
better than its scores claim, so calibration fitted on the calibration slice
does not transfer perfectly to the test period. Recorded rather than papered
over: on this rail, CBPE currently understates performance and should be read as
a *change detector*, not as an absolute number.

## Tested against simulated drift

| Scenario | Worst severity | What fired |
| --- | --- | --- |
| No drift (two halves of one period) | **OK** | nothing |
| Scores shifted +40% | **CRITICAL** | alert rate 2.00% -> 4.08% |
| Amounts and counts doubled | **WATCH** | C13 PSI 3.81 |

## Two bugs the tests caught

Both would have made the monitor useless by crying wolf every single day.

**1. Time-index features always "drift".** `DT_M` is a month index, so it
necessarily differs between one period and the next. It scored a PSI of **5.14**
on two halves of the same period with no drift at all - twenty times the "major"
line. Time-index features are now excluded by name.

**2. Reconstructing a distribution from quantiles is wrong for concentrated
features.** The baseline stored eleven quantiles and rebuilt the reference by
interpolating between them. Fine for a smooth feature; badly wrong for the C
columns, which are counts concentrated on small integers - interpolation turns a
spike at zero into a continuous ramp that nothing matches.

The scale of the error: with reconstruction, **8 of 20 features** exceeded the
major threshold on data with no drift, `P_emaildomain_count` scoring 3.60 and
`C9` 2.37. Storing the actual histogram bins instead takes that to **0 of 16**.

## Rollback

`ModelRegistry` versions every model with the metrics it was accepted on.

```
--- a WORSE retrain ---
{'promoted': False, 'reason': 'recall_at_operating_point would fall 18.4%
 (0.4780 -> 0.3900), over the 5% tolerance', 'active': 'v2'}

--- forced anyway, then rolled back ---
{'promoted': True, 'active': 'v3', 'reason': 'forced'}
{'rolled_back': True, 'from': 'v3', 'to': 'v2'}
```

**The gate compares recall at the operating point, not PR-AUC.** Measured on the
wire rail across three seeds: PR-AUC varied **20.5%** while recall at 1% varied
**0.9%**. Gating on the noisier metric would reject good retrains and accept bad
ones at random.

A 5% tolerance allows normal retrain-to-retrain variation while catching a real
regression. `force=True` exists because a human sometimes knows something the
metrics do not - and `rollback` exists for when they were wrong.
