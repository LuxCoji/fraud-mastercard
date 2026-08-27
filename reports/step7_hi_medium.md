# HI-Medium - recall up 27%, instability unchanged

HI-Small produced a working wire model whose numbers could not be trusted:
PR-AUC +-0.0292 across three seeds (19.4% relative), early stopping anywhere
between 75 and 588 rounds, and an adoption verdict that passed its own test by
0.0002. The stated cause was the positive count - 3,380 laundering cases in four
million rows, so which subset bagging draws matters enormously.

HI-Medium tests that: 31.9M rows and 2,756 labelled rings against 5.1M and 370.

## Results

| | HI-Small | HI-Medium |
| --- | --- | --- |
| Rows | 5,078,345 | 31,898,239 |
| Laundering in fit slice | 3,380 | 20,163 |
| **PR-AUC** | 0.1508 +-0.0292 | **0.3012 +-0.0617** |
| **Recall at 1%** | 54.18% | **68.65% +-0.62%** |
| **Seed spread (PR-AUC)** | 19.4% | **20.5%** |
| Rounds chosen | 399 / 92 / 153 | 351 / 411 / 357 |

Per seed: PR-AUC 0.3254, 0.3471, 0.2311. Recall at 1%: 69.24%, 68.70%, 68.01%.

## The explanation was half right

**Recall improved sharply.** 54.18% -> 68.65% at the same 1% alert rate, an
extra 14.5 points of laundering caught. PR-AUC doubled. Six times the positives
bought a materially better model, exactly as expected.

**The instability did not go away.** The seed spread on PR-AUC is 20.5%, against
19.4% before - unchanged, or fractionally worse. So the sample-size explanation
does not survive contact with the data.

But look at which metric is unstable:

| Metric | Spread |
| --- | --- |
| PR-AUC | **20.5%** |
| **Recall at 1%** | **0.9%** (68.01-69.24%) |

**Recall at the operating point is now rock steady.** The volatility is confined
to PR-AUC, which integrates over every alert rate including the 50% end nobody
will ever run. Seed 2024 scored 0.2311 against seed 7's 0.3471 - a 50% gap on
PR-AUC - while their recall differed by 0.7 points.

That is a much better-behaved model than the headline spread suggests, and it
vindicates the earlier decision to adopt XGBoost on recall rather than on a
PR-AUC verdict that passed by 0.0002. **Recall at the operating point is the
metric this rail should be judged and reported on.** Round counts agree:
351/411/357 here against 399/92/153 before.

## Cycle enumeration hit its budget

```
graph: 1,429,365 nodes, 3,541,221 edges
cycle enumeration stopped: 900s budget reached after 1,200,000 cycles
```

The features are therefore built on a **partial** cycle count. Two consequences:

- These numbers are a **floor**, not a ceiling. A longer budget would give
  richer cycle features.
- The time budget was necessary. The previous run reached the identical graph
  and then produced nothing for 52 minutes before cancellation, because the
  2,000,000-cycle cap only fires when a cycle is *yielded* and the search
  between yields was the cost.

Feature building took **207 minutes** against under a minute for HI-Small.
Cycle enumeration is the entire cost and it scales far worse than the data does.

## HI-Large is not worth attempting

- Feature building is already 3.5 hours at 32M rows; HI-Large is 17 GB.
- IBM's own results show gradient boosting *degrading* there - XGBoost 65.7% on
  Medium against 42.7% on Large.
- Cycle enumeration would hit its budget far earlier, so the features would be a
  smaller fraction of the true counts than they already are.

**HI-Medium is the right dataset for this rail.**

## What changes

The wire rail moves to HI-Medium: **68.65% of laundering caught at a 1% alert
rate**, up from 54.18%. Reported with a seed spread on recall (+-0.62%) rather
than on PR-AUC, because that is what the rail is operated on.

---

# Artifacts exported and promoted

The first HI-Medium run measured 68.65% and saved nothing, so the shipped model
stayed the 55.48% HI-Small one. Re-run with an export step:

```
EXPORTED  recall@1% 69.48%  PR-AUC 0.3262
```

| | HI-Small (v1) | **HI-Medium (v2)** |
| --- | --- | --- |
| Rows | 5,078,345 | 31,898,239 |
| Labelled rings | 370 | 2,756 |
| **Recall at 1%** | 55.48% | **69.48%** |
| PR-AUC | 0.1377 | 0.3262 |
| Precision at 1% | ~10% | 11.6% |
| Brier | - | 0.00137 |

**+14 points of laundering caught at the same alert rate.**

Registered both and promoted through the gate rather than by overwriting a file:

```
promote v2 -> {'promoted': True, 'active': 'v2', 'previous': 'v1',
               'reason': 'metrics acceptable'}

rail    version   recall@op   active
wire    v1         0.5548
wire    v2         0.6948      <--
```

v1 is retained, so `registry.rollback("wire")` restores it in one call if the
new model misbehaves in a way the held-out metrics did not show.

## The seed spread got worse, and the reason is one seed

| | run 1 | run 2 |
| --- | --- | --- |
| PR-AUC spread | 20.5% | **36.7%** |
| Recall at 1% spread | 0.62% | **1.18%** |

Seed 2024 scored PR-AUC 0.1606 against seed 42's 0.3262 - a 2x gap - while their
recall at 1% differed by 2.3 points (67.20% vs 69.48%).

That is the same pattern seen throughout this rail, now in a sharper form:
**PR-AUC is unstable here and recall at the operating point is not.** Reporting
PR-AUC for the wire rail invites exactly the wrong conclusion; recall at 1% is
the number to quote, and it should carry its spread.

## Standing caveats

- Cycle enumeration hit its 900-second budget after 1.2M cycles, so the pattern
  features are built on **partial** counts. 69.48% is a floor.
- The source is synthetic. Checked clean of the artefact that made PaySim
  unusable, but clean is not real.
