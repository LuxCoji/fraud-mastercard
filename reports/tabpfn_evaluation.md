# TabPFN-2.5 evaluation - parked, not adopted

Tabular foundation models are reported to beat tuned tree models on TabArena,
with an 87% win rate on datasets up to 100K rows, and to adapt to new fraud
families in-context without retraining. Worth testing. Tested, and parked.

## The constraint that shaped everything

**TabPFN accepts at most 50,000 training rows.** Our card training set is
472,432. The paper is explicit that scaling to millions is future work.

So the comparison had to be constructed: both models see the same 50,000-row
subsample (every fraud row that fits plus a random sample of legitimate rows),
scored on the same untouched chronological test set.

## What the handicap costs, measured

| Model | Rows seen | PR-AUC | Fraud caught at 2% |
| --- | --- | --- | --- |
| XGBoost | 50,000 | 0.5608 | 44.61% |
| **XGBoost** | **472,432** | **0.5735** | **45.84%** |

**Using all the data is worth about 2% PR-AUC.** That is the bar TabPFN had to
clear while permanently blind to 90% of the training set - it needed to beat
XGBoost-on-50k by more than the amount XGBoost gains from seeing everything.

## It never ran. Three environments, three unrelated failures

| Environment | Failure |
| --- | --- |
| Local CPU | Refuses above 5,000 rows without a GPU - a second, tighter cap |
| Local, browser auth | `select.select()` on stdin, which does not work on Windows; their manual-paste fallback crashes with `WinError 10038` |
| Kaggle T4 | `CUDA error: no kernel image is available for execution on the device` - their PyTorch build does not match the GPU architecture |

The licence and token problems were solved along the way (`TABPFN_TOKEN` plus
`TABPFN_NO_BROWSER=1` authenticates without a browser). The CUDA mismatch is in
their packaging.

## Decision

**Parked.** Not because the idea is wrong - the in-context learning argument is
genuinely interesting, and adapting to five new fraud examples with no retraining
is something gradient boosting cannot do at all. But:

- Three environments, three unrelated failures, is a maturity signal.
- Even working, the 50K cap is permanent and we measured what it costs.
- The XGBoost baselines it needed to beat are recorded above, so revisiting this
  later is cheap.

## What would change the answer

Support for datasets in the millions, which the authors say is in progress. At
that point the row cap disappears and the in-context adaptation argument stands
on its own rather than being offset by a data handicap.

The evaluation code and the 33 MB comparison dataset are kept out of the project
tree - in the session scratchpad and as a private Kaggle dataset
(`onethybeing/card-fraud-tabpfn-eval`) - so nothing here affects the shipped
pipeline.
