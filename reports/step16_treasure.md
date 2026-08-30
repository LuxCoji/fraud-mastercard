# Step 16 - TREASURE on real data: rejected as a replacement, but it found something

MCB's reimplementation of Visa's TREASURE (arXiv:2511.19693) was rejected once
for its **data** - `data_gen.py` wrote the label, so PR-AUC 0.9989 measured the
generator, not the model. The architecture was never the problem. This runs the
same architecture on IEEE-CIS: real transactions, real labels, the same held-out
final 42 days the gradient boosted model is measured on.

## Result

| | recall@2% | PR-AUC | AUC |
| --- | --- | --- | --- |
| LightGBM, 292 features | **45.19%** | **0.5865** | 0.924 |
| TREASURE, sequence | 31.97% | 0.3849 | 0.851 |

Three seeds: 31.82 / 31.08 / 33.02. Spread 0.0101, gap **-13.22%**.

**Rejected as a replacement** under the rule fixed before the run: beat recall at
the 2% operating point by more than the seed spread. It does not.

## But it is a real model, and it never saw a V column

This is the part worth keeping. TREASURE was given:

- **static**: card1-6, addr1, addr2
- **dynamic**: amount, time since the card's last transaction, dist1-2, C1-C14,
  D1-D15, ProductCD, email domains, DeviceType, hour, weekday, M1-M9

**No V columns at all.** `V258` and `V257` are the two strongest features in the
gradient boosted model, and TREASURE reached AUC 0.851 without them.

Set that against the measurement taken when the FinSim integration was scoped -
the gradient boosted model, restricted to what a payment simulator can actually
supply:

| Model | V columns | recall@2% |
| --- | --- | --- |
| LightGBM, everything | yes | **45.3%** |
| **TREASURE, sequence** | **no** | **32.0%** |
| LightGBM, FinSim-supplyable features | no | **14.1%** |

**Without the anonymised columns, the sequence model more than doubles what the
per-row model achieves.** Reading a card's trajectory recovers signal that
per-row features cannot, which is exactly the claim the paper makes and exactly
what MCB's synthetic data could not test.

## What this does not yet prove

TREASURE still used `C1-C14` and `D1-D15`. Those are counts and time deltas -
the kind of thing computable from any raw transaction stream, unlike the V block
which is Vesta's own engineered output and unrecoverable elsewhere. **But that is
an argument, not a measurement.** The decisive experiment is TREASURE restricted
to fields a payment simulator genuinely emits: amount, time delta, hour, weekday,
card identity, counterparty. That has not been run.

## The scale caveat, restated

TREASURE was trained on **six billion** transactions from 30 million cardholders.
This ran on 590,540 from 217,850 - four orders of magnitude less, against an
architecture whose own paper reports performance improving with scale. The median
card here has **one** transaction, so on a third of the book the sequence model
has no trajectory to read at all. 31.97% is what this architecture does on a
small dataset, not what it does.

## A bug that cost a full run, and was visible in the output

The first completed run reported **AUC 0.5007 across all three seeds** - exactly
chance. That was not a weak model; it was a broken one.

`prepare()` sorted the frame but kept the original index labels, while the caller
read scores back positionally with `.to_numpy()[row_index]`. After a sort, labels
and positions differ, so every score was matched to a different transaction's
label.

The evidence was printed and went unread:

```
train 472,432  test 118,108 (4,064 fraud)     -> 3.44%
scored rows 118,108  fraud 3.915%             -> 4,624
```

Two different fraud counts for the same 118,108 rows. The alignment is now
asserted - row count, fraud count, and set equality of the row indices - because
this class of bug is silent: the model trains, the loss falls, and only the
metric is wrong.

Three earlier runs failed on GPU environment issues rather than anything in the
model: Kaggle allocated a Tesla P100 (compute 6.0) against a PyTorch built for
sm_70 and above. The kernel now reads the compute capability from `nvidia-smi`
and installs a matching build **before** the first torch import, since a loaded C
extension cannot be replaced in-process.
