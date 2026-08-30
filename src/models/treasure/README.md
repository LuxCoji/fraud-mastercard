# TREASURE on real data

`model.py` is Visa's TREASURE architecture, taken unchanged from the MCB
reimplementation (arXiv:2511.19693). It was not the part that failed.

What failed was MCB's **data**: `data_gen.py` injected attacks with fixed
signatures and labelled exactly those rows, so the model recovered its own
generator's if-statements and reported PR-AUC 0.9989. Its next-merchant
prediction scored 2.26e-05 against a 2.0e-05 random baseline - the sequence
modelling, which is the entire justification for a transformer here, produced
nothing measurable.

`sequences.py` puts the same architecture on IEEE-CIS: real transactions, real
labels, a chronological split, and the same held-out 42 days the gradient
boosted model is measured on.
