# TREASURE - evaluated, not adopted

A reimplementation of Visa's TREASURE payment foundation model (arXiv:2511.19693),
supplied as `MCB.zip`. Reviewed and rejected. Two artifacts kept.

## Why it was rejected

Its reported numbers are **PR-AUC 0.9989, F1 0.956** on its own test split. Those
are not performance figures.

**The generator writes the label.** From its own README:

> Every fraudulent transaction is labelled `abnormal_flag = 1` - TREASURE's
> supervision.

`data_gen.py` injects attacks with fixed signatures - card testing is a burst of
tiny auths, account takeover is foreign geography plus a high amount - and marks
exactly those rows as the target. The model recovers the generator's own
if-statements. This is the same circularity that made PaySim unusable here, and
it is the failure the previous project's `DATA_SOURCES.md` warned about.

**The sequence modelling did not work.** Next-merchant prediction scores
**2.26e-05**. Random guessing over ~50,000 merchants scores **2.0e-05**. That is
chance. The entire justification for a sequence transformer - reading a
cardholder's trajectory to catch sleepers and low-and-slow attacks - produced
nothing measurable. What the model learned was the label rule, which needs no
transformer.

Against our card rail at PR-AUC 0.578 on real IEEE-CIS data: a lower honest
number beats a higher fabricated one.

## What was worth keeping

**`dashboard.html`** - a working fraud console. We need one, and this is a
reasonable starting layout. It also discloses its synthetic limitation *in the
UI*, which is the right instinct and worth copying.

**The paper** - TREASURE itself is legitimate Visa research. The architecture is
sound; the reimplementation's data is what fails. Worth reading before any
sequence-model work on real data.

**The attack taxonomy** - card testing / account takeover / behavioural shift /
mule is a sensible vocabulary for describing what a system catches.

## What was discarded

The trained checkpoint (68 MB), the synthetic dataset, and the training code.
None can inform a decision about a model that runs on real transactions.
