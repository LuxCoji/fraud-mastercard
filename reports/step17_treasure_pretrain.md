# Step 17 - TREASURE, one lever at a time: pretraining is the one that works

Step 16 measured this architecture at **31.97% recall at a 2% flag rate** and
rejected it as a replacement for the gradient boosted card model. That run had
the next-transaction head switched off - the self-supervised objective the paper
is built on. This turns it back on and measures five variants, each adding
exactly one thing to the one before, so a gain can be attributed rather than
guessed at.

Three seeds per arm. Same chronological split as every other card measurement:
the final 42 days held out, no statistic fitted across the boundary.

## Result

| arm | recall@2% | precision@2% | genuine stopped | PR-AUC | AUC |
| --- | --- | --- | --- | --- | --- |
| C0 signal head only, 8 epochs | 32.37% | 55.7% | 1,047 | 0.3887 | 0.854 |
| C1 + next-transaction head | 32.50% | 55.9% | 1,041 | 0.3941 | 0.854 |
| C2 + 20 epochs | 32.94% | 56.7% | 1,023 | 0.4035 | 0.865 |
| **C3 + self-supervised pretraining** | **33.74%** | **58.0%** | **991** | **0.4093** | 0.851 |
| C4 + a larger model | 32.51% | 55.9% | 1,041 | 0.3889 | 0.838 |

Decision rule, fixed before the run: beat C0's recall at the 2% operating point
by more than the combined seed spread, **and** win on every individual seed.

| arm | gap | spread | every seed | verdict |
| --- | --- | --- | --- | --- |
| C1 | +0.13% | 0.0047 | no | not proven |
| C2 | +0.57% | 0.0065 | yes | not proven |
| **C3** | **+1.37%** | **0.0119** | **yes** | **adopt** |
| C4 | +0.15% | 0.0108 | no | not proven |

## What that means, and what it does not

**Pretraining is the lever.** Running the next-transaction objective *first*, with
no labels at all, and only then training on fraud, is worth +1.37 points. Adding
the same objective as a joint auxiliary loss without a separate pretraining
phase (C1) is worth nothing measurable.

At a fixed 2% flag rate the trade is one for one: C3 catches **55 more frauds and
stops 55 fewer genuine customers** per 118,108 transactions than C0. Recall and
false positives improve together because the flag budget is held constant.

**The margin is thin.** The gap is 1.37% against a combined spread of 1.19%. It
clears the rule, but by less than a fifth of the spread, and the per-seed
picture shows why that caution is warranted:

| seed | C0 | C3 |
| --- | --- | --- |
| 42 | 32.87% | 34.06% |
| 7 | 32.06% | 34.65% |
| 2024 | 32.16% | 32.50% |

Seed 2024 wins by 0.34 points. C3 is adopted because it satisfies a rule fixed
in advance, not because the effect is large.

**A bigger model made it worse.** C4 doubled the parameter count and lost 1.2
points against C3. The paper reports performance improving with model size on
six billion transactions; on 590,540 the same change overfits. That is a
consistent result rather than a contradictory one, and it closes off "just make
it bigger" as a direction here.

## The lever that cannot be pulled on this dataset

The paper's headline mechanism is self-supervised pretraining on an enormous
unlabelled corpus. IEEE-CIS ships one - `test_transaction.csv`, roughly 506,000
rows with no labels - and it is **unusable here**. Those rows fall entirely
after `train_transaction.csv`, so they are after the held-out window too.
Pretraining on them would let the model read the future, which is the one thing
a chronological split exists to prevent. Any recall it produced would not be
achievable in production.

So the +1.37 points measured above come from pretraining on the *training rows
themselves*, seen twice under two different objectives. That is the weakest
possible version of the mechanism, and it still cleared the bar.

**FinSim removes this constraint.** The simulator generates unbounded unlabelled
traffic, and by training time all of it is genuinely in the past. The lever that
worked here at its weakest can be pulled properly there. That is a prediction to
measure inside the sandbox, not a claim about what the number will be.

## Round two: does more pretraining pay, and what do the simulator fields cost?

Round one left two questions open. Round two answers both, with C3 re-run as its
own control.

| arm | recall@2% | spread | genuine stopped | PR-AUC |
| --- | --- | --- | --- | --- |
| C3, pretrain 6 (control) | 33.74% | 0.0111 | 991 | 0.4093 |
| C5, pretrain 12 | 32.50% | 0.0047 | 1,041 | 0.3720 |
| **SIM, simulator fields only** | **33.23%** | 0.0099 | 1,011 | 0.3979 |

**More pretraining does not keep paying.** Twelve epochs scored 1.24 points
*below* six, losing on the mean and on two of three seeds. Six is the setting.

**The simulator fields cost almost nothing.** Restricting the model to what
FinSim can emit - no billing address, no email domain, no geolocation, none of
Vesta's M flags - costs 0.50 points against a combined spread of 1.48. The two
are not distinguishable at three seeds.

The control reproduced round one seed for seed (34.06 / 34.65 / 32.50), so the
pipeline is deterministic and these are not lucky draws.

Round one's simulator arm was run on **C4's recipe**, which the same run showed
to be the worse one. Its 30.31% understated the result by nearly three points
and should not be quoted; 33.23% is the number.

That correction changes the comparison that governs the integration:

| model | anonymised V columns | recall@2% |
| --- | --- | --- |
| LightGBM, all 292 features | yes | 45.3% |
| **TREASURE, simulator fields only** | **no** | **33.2%** |
| LightGBM, simulator-supplyable features | no | 14.1% |

On the fields a payment simulator can supply, the sequence model scores **2.4
times** the per-row model, and gives up almost nothing against its own
full-field version.

## What round two settles, and what it leaves open

Settled: pretrain six epochs, fine-tune twenty, standard model size, simulator
fields. That recipe is the one to ship.

Open, and measured next: `card1` and `card2` are still in the inputs, and they
are identifier-like - 13,553 and roughly 500 distinct values. The model learns
one embedding per value, and an embedding for card1=13926 means nothing in
another payment system. If they matter, the weights cannot leave this dataset
and the sandbox model must be trained from scratch on simulator traffic. If they
do not, the pretrained weights transfer and only fine-tuning is needed.

The paper does not use card identifiers at all - it groups transactions by card
and feeds the model card properties, handling even 150-million-cardinality
merchants through category mappings rather than ID embeddings. So the
expectation is that dropping them costs little. That is an expectation, not a
measurement, which is why it is being measured.

## A crash the smoke test caught before Kaggle did

A batch in which every card has exactly one transaction has no position-to-
position transition, so the self-supervised loss has nothing to compute.
Returning a plain zero there detaches the loss from the autograd graph and
`backward()` raises `element 0 of tensors does not require grad`.

The median card in IEEE-CIS has one transaction. That is the data's most common
shape, not an edge case, and it would have killed the run in the first epoch. A
thirty-second local run on 400 synthetic rows found it; two earlier Kaggle runs
were lost to bugs of exactly this kind that no local check was written for.

## Files

- `src/models/treasure/train.py` - both objectives, the pretraining phase, `Arm`
- `src/models/treasure/sequences.py` - `SIM_*` column sets
- `src/models/treasure/config.py`, `model.py` - the architecture, vendored from
  the MCB reimplementation of arXiv:2511.19693
- `tests/test_regressions.py` - four tests covering the above
