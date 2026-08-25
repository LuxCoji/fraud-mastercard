# Step 4 Results - Wire rail (IBM AML)

Measured on the held-out final 10 days: 1,015,669 transactions, 1,797
laundering, 0.1769% base rate. Rows sorted by time before splitting - the file
is not in chronological order. All account statistics fitted on the training
period only.

**Standing caveat: this source is synthetic.** It was checked for the artefact
that made PaySim unusable and is clean - its strongest single signal (ACH
format) covers 86.6% of laundering but also 11.8% of legitimate traffic, a
correlation rather than a giveaway. Clean is not real. These numbers are
plausible; they are not proven the way the card rail's are.

## Headline

| | Logistic (floor) | LightGBM on graph features |
| --- | --- | --- |
| PR-AUC | 0.0196 | **0.1048** |
| AUC | - | 0.9756 |
| Brier | 0.001748 | 0.001762 |
| Brier baseline | 0.001766 | 0.001766 |

**5.35x the floor.** For scale, a random model scores PR-AUC at the base rate,
0.00177 - so the logistic floor is already 11x random and LightGBM is 59x.

## The graph framing earns its keep

**Account-graph features carry 63.7% of total gain.** Nine of the top fifteen
features are account-level rather than row-level:

| Feature | The shape it detects |
| --- | --- |
| `sent_total`, `recv_total` | volume through an account |
| `sent_partner_ratio` | FAN-OUT, SCATTER - many one-off counterparties |
| `recv_partner_ratio` | FAN-IN, GATHER |
| `receiver_passthrough` | STACK / layering - money in matches money out |
| `amount_vs_sender_mean` | a transaction unlike the account's own history |

That was the bet of this rail: laundering is a topology, and the 370 labelled
rings are CYCLE, FAN-IN, FAN-OUT, GATHER-SCATTER, SCATTER-GATHER, BIPARTITE,
STACK and RANDOM. None of those exists inside a single row. The measurement
supports the bet.

## Operating points

| Flag rate | Laundering caught | Precision | Legitimate hit | False alarms per catch |
| --- | --- | --- | --- | --- |
| 0.1% | 10.9% | 19.2% | 0.08% | 4.2 |
| 0.5% | 28.6% | 10.1% | 0.45% | 8.9 |
| **1.0%** | **47.7%** | 8.4% | 0.92% | **10.9** |
| 2.0% | 73.6% | 6.5% | 1.87% | 14.4 |
| 3.0% | 82.6% | 4.9% | 2.86% | 19.5 |
| 5.0% | 88.1% | 3.1% | 4.85% | 31.1 |

## This rail cannot use the card rail's policy

Precision peaks around 19% and sits near 6-8% at usable recall. On the card rail
the same 2% flag rate gives 78% precision. The difference is not model quality -
it is the base rate, 0.177% against 3.44%, nearly twenty-fold.

**So the wire rail must not auto-block.** Fourteen legitimate wires stopped for
every laundering case caught would be indefensible. The correct consumer here is
a **human investigator queue**, which is how AML actually works in banks - alert
volumes are sized to analyst capacity and single-digit precision is normal and
accepted.

The card rail decides. The wire rail *recommends*. Same codebase, different
economics, and the policy layer has to encode that difference rather than share
one threshold.

## AUC 0.9756 next to PR-AUC 0.1048

Worth stating plainly, because the first number is the flattering one and it is
the one a stakeholder will quote. At a 0.177% base rate, AUC is dominated by how
well the model orders the overwhelming majority of legitimate traffic, which is
easy. PR-AUC asks how the top of the ranking looks, which is what an
investigator actually sees.

**AUC 0.9756 is real and nearly useless. PR-AUC 0.1048 is the number that
governs.** This is exactly the trap accuracy sets on imbalanced data, one metric
further along.

## Calibration is marginal

Brier 0.001762 against a baseline of 0.001766 - technically better, by 0.2%.
The card rail's blend beat its baseline by 30%. Before this rail drives any
probability threshold it needs the same Platt layer the card rail got; the
current scores are usable for ranking only.

## Not done

- Calibration layer (above).
- Ring-level evaluation. Every number here is per-transaction, but the ground
  truth is 370 *rings*. Catching one transaction in a cycle is worth more than
  the count suggests, and catching all eight legs is worth more again. A
  ring-level recall metric would measure what an investigator cares about.
- The eight topology labels are unused as features or as evaluation strata.
