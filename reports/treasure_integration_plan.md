# TREASURE on real data - research, test, and integration plan

## What was reviewed

| Source | What it is |
| --- | --- |
| `MCB (2).zip` | A reimplementation of TREASURE, with model code and a trained checkpoint |
| arXiv:2511.19693 | The Visa paper itself - the same PDF ships inside the zip |
| VVAH | Visa's vulnerability agentic harness - already evaluated in step 13 |

## The distinction that matters

MCB was rejected once, and the reason was **its data, not its architecture**.

Its `data_gen.py` injects attacks with fixed signatures - card testing is a burst
of tiny authorisations, account takeover is foreign geography plus a high amount
- and labels exactly those rows. The model recovers the generator's own
if-statements and reports **PR-AUC 0.9989**. Its next-merchant prediction scored
**2.26e-05** against a **2.0e-05** random baseline, which is chance: the sequence
modelling, the entire justification for using a transformer, produced nothing
measurable.

Reading the code confirms the architecture is a faithful, clean implementation
of the paper. So the question worth asking is not "is MCB any good" but **"does
TREASURE's architecture beat gradient boosting on real data?"**

## The architecture

A small causal transformer over one cardholder's history:

```
static attributes  ->  InputModule  ->  position 0
each transaction   ->  InputModule  ->  positions 1..t
                          |
              3-layer Transformer decoder, 4 heads, d_model 256
                          |
          +---------------+---------------+
   next-transaction head          network-signal head
   (16 attributes, self-supervised)   (includes the fraud flag)
```

Two things make it different from our gradient boosted model:

- It reads a **sequence**, so a transaction that is unremarkable alone can be
  flagged as the third of five rapid tests on a stolen card.
- The next-transaction head is **self-supervised** - it learns from the shape of
  normal behaviour without needing fraud labels, which is what makes it a
  foundation model rather than a classifier.

## What IEEE-CIS can and cannot support

Measured, not assumed:

| | |
| --- | --- |
| Cards (UIDs) | 217,850 |
| Transactions | 590,540 |
| **Median transactions per card** | **1** |
| Cards with >= 3 transactions | 26.0%, covering **66.6% of traffic** |
| Fraud rate on those cards | **3.92%** |
| Fraud rate on cards with < 3 | 2.65% |

**There is a real sequence signal to find** - two thirds of traffic sits on cards
with history, and those cards carry proportionally more fraud. But a third of the
book has no history at all, and on those rows a sequence model sees exactly what
a per-row model sees.

**The scale gap is the honest caveat.** TREASURE was trained on **six billion**
transactions from 30 million cardholders over 26 months. IEEE-CIS has 590,540
from 217,850 over six. That is four orders of magnitude less data, against an
architecture whose own paper (section 4.7) reports that performance improves with
scale. A weak result here would be evidence about this dataset, not about
TREASURE.

## The mapping, stated exactly

The paper cannot disclose its attribute names. IEEE-CIS is public, so ours can be:

| Paper | IEEE-CIS |
| --- | --- |
| 5 static (per-card, constant) | card1-6, addr1, addr2 |
| 16 dynamic numerical | TransactionAmt, time since last txn, dist1-2, C1-C14, D1-D15 |
| dynamic categorical | ProductCD, P/R_emaildomain, DeviceType, hour, weekday, M1-M9 |
| 2 network signals | isFraud |

Card identity is the same UID the gradient boosted model uses: `card1 + addr1 +
first-seen day`, where first-seen day is `day - D1`.

## The experiment now running

Kaggle, GPU: https://www.kaggle.com/code/onethybeing/treasure-ieee-cis

Three seeds, the **same chronological split** the gradient boosted model uses -
final 42 days held out, no statistic fitted across the boundary, categorical
codes fitted on training rows only so an unseen category cannot borrow another's
embedding.

**Decision rule, fixed before results:**

```
LightGBM, three seeds:   recall@2%  45.19%    PR-AUC 0.5865  (spread +-0.0024)

Adopt TREASURE only if recall at the 2% operating point beats that by more
than the combined seed spread.
```

Recall decides, not PR-AUC. That rule comes from the wire rail, where PR-AUC
varied 36.7% across seeds while recall varied 1.2% - gating on PR-AUC would
accept and reject models at random.

## Integration plan, conditional on the result

### If TREASURE wins outright

1. Keep both models. The router calls TREASURE where the card has history and
   the gradient boosted model everywhere else - which is 33% of traffic.
2. Latency must be re-measured. The current p99 is 5.53ms for two boosted
   models; a transformer forward pass is a different cost, and the 100ms
   authorisation budget is the constraint.
3. `FastCardScorer`'s dictionary-lookup design does not transfer - a sequence
   model needs the card's recent history at scoring time, which means a feature
   store, not a lookup table.

### If TREASURE loses but its embeddings help

The paper's own suggested use is TREASURE as a **feature extractor**. Take the
hidden state and feed it to LightGBM alongside the existing 292 features. This is
the more likely outcome and the cheaper integration: no second serving path, and
the embedding is computed once per transaction.

Same decision rule applies. **It has to clear the seed spread.**

### If neither helps

Record it and stop. That would be the fifth measured rejection on this project,
and the reason would be worth stating: with a median of one transaction per card,
there may simply be no trajectory to read.

## Risks

- **Scale.** Four orders of magnitude below the paper's training set. A null
  result is about IEEE-CIS, not about the architecture.
- **Class imbalance.** Fraud is 3.5%. The runner uses a class weight inside the
  loss, which is a weighting, not resampling - the project bans `scale_pos_weight`,
  SMOTE and `class_weight="balanced"` on the boosted models for reasons measured
  earlier, and a regression test enforces it.
- **Sequence leakage.** The causal mask means position t sees only 0..t. Worth a
  test rather than a reading of the code.
- **Latency.** Unmeasured. It decides whether this can serve at all, regardless
  of accuracy.
