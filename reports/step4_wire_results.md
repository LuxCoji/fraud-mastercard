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

---

# Pattern counts - the largest single improvement in the project

The first version of this rail counted what an account does in aggregate:
degrees, totals, pass-through ratios. The IBM paper behind this dataset
attributes its strong gradient-boosted result to counting the laundering
*shapes* themselves. Measured on our split:

| | Aggregates only | + pattern counts |
| --- | --- | --- |
| Features | 25 | 49 |
| **PR-AUC** | 0.1048 | **0.1522** |
| AUC | 0.9756 | 0.9706 |
| Brier | 0.001762 | **0.001654** |
| Rounds chosen | 145 | 618 |

**+45.2% PR-AUC.** Against a 0.4% noise floor, that is not close.

Pattern-count features carry **30.7% of total gain**, with `from_two_hop_payers`
and `to_two_hop_payers` both in the top twelve.

## Where the gain actually lands

| Flag rate | Aggregates | + patterns | Change |
| --- | --- | --- | --- |
| 0.1% | 10.9% | **14.5%** | +3.6 |
| 0.5% | 28.6% | **33.7%** | +5.1 |
| 1.0% | 47.7% | **48.4%** | +0.7 |
| 2.0% | **73.6%** | 67.3% | **-6.3** |
| 3.0% | **82.6%** | 75.2% | **-7.4** |

The improvement is concentrated at the **top of the ranking** and reverses
below 2%. Precision at 0.1% rises from 19.2% to 25.6%, and false alarms per
catch fall from 4.2 to 2.9.

That is the right shape for this rail. An investigator works the top of a
queue, so a model that is sharper in its first few hundred alerts and blunter in
its thousandth is more useful than the reverse - even though the aggregate-only
model catches more laundering if you are willing to review 3% of all wires,
which no bank is.

## The time windows were the fix

The failed standalone cycle detection used a 30-day window on an 18-day dataset,
so every cycle trivially passed and 71.6% of all accounts were flagged. The
paper's windows are **one day for cycles, six hours for scatter-gather**. Same
algorithm, useful output.

## Two patterns still uncovered

Of the eight labelled shapes, five are counted directly (CYCLE, FAN-IN,
FAN-OUT, SCATTER-GATHER, GATHER-SCATTER) and three are not:

- **STACK** (43 rings) - a chain A->B->C->D->E. Detected only indirectly through
  pass-through ratios; nothing traces the chain itself.
- **BIPARTITE** (49 rings) - one group paying another group. Approximated by
  two-hop neighbour counts, which is a hint rather than the structure. Notably
  those two-hop features are the strongest of the new set, which suggests
  building the structure properly would pay.
- **RANDOM** (41 rings) - has no shape by construction. No graph feature can
  address it; only amount and timing anomalies can.

## AUC fell while PR-AUC rose

0.9756 -> 0.9706 on AUC, 0.1048 -> 0.1522 on PR-AUC. Both are correct. AUC
rewards ordering the easy majority of legitimate traffic; PR-AUC rewards the top
of the ranking. The pattern features traded a little of the former for a lot of
the latter, which is the trade this rail wants.

---

# Chain and bipartite detectors - rejected

`from_two_hop_payers` and `to_two_hop_payers` - crude stand-ins for bipartite
structure - were among the strongest of the pattern features, which suggested
building the real structure would pay. It did not.

| | Pattern counts | + chain & bipartite |
| --- | --- | --- |
| Features | 49 | 67 |
| **PR-AUC** | **0.1522** | 0.1416 |
| AUC | 0.9706 | 0.9735 |
| Rounds chosen | 618 | **116** |

**-7.0%.** Rejected.

## Why, most likely

The tree count is the tell: 618 rounds down to 116. The model stopped improving
five times sooner with more features available, which is what happens when new
columns are noisy enough to produce splits that do not generalise. Eighteen
weakly-informative features across 49 existing ones is enough to make early
stopping fire early.

## The detector was measured against ground truth first, and was weak

Tested on the labelled rings before being wired in:

| Shape | Rings found by the chain detector |
| --- | --- |
| CYCLE | 25 / 54 |
| RANDOM | 19 / 41 |
| **STACK** | **7 / 43** |
| GATHER-SCATTER | 5 / 51 |
| **BIPARTITE** | **4 / 49** |
| FAN-OUT, FAN-IN | 4 / 48, 4 / 40 |
| SCATTER-GATHER | 1 / 44 |

It finds cycles better than the stacks it was written for, and the bipartite
detector proper found 1 ring in 49.

Two real bugs were found and fixed by measuring the ground truth first, and
neither was enough:

- The time window was 72 hours. Labelled STACK rings span a **median of 73
  hours** - the round number excluded more than half of them by an hour. Widened
  to 120.
- The value band was 0.70-1.10. Real hop ratios have a median of 0.99 but a p10
  of 0.05 and p90 of 28.7, so the tight band admitted only 34.8% of genuine
  hops. Widened to 0.10-10.0.

Both fixes together took chains found from 120 to 720 and ring coverage from
5/43 to 7/43. The approach is wrong, not the parameters.

## Why forward path-walking is the wrong tool here

A STACK is a chain of *transactions*, but the detector walks a graph of
*collapsed account pairs* - parallel transactions between one pair are summed
into a single edge. That destroys exactly the sequence a layering chain is made
of. A correct STACK detector would walk transactions in time order, not edges.

BIPARTITE fails differently: it is a property of a *set* of accounts, and this
implementation asks a per-account question ("do my payees have other payers?").
The right formulation is dense-subgraph or biclique detection over the whole
graph, which is a different and much more expensive algorithm.

## Kept anyway, for a different job

`chains.py` is retained and wired into `policy/cases.py` rather than into the
feature set. It is a poor *feature* but a working *explanation*: given an
account the model has already flagged, it returns the actual path -

```
811A67510 -> 812D1B280 -> 811FCA7B0 -> 811EDA940 -> 811E3C9F0 -> 81235EC40
7 hops, 49,168 in / 37,845 out (77% retained) over 37h
```

That is the difference between an alert and a case, and it does not need to
improve PR-AUC to be worth having. The cost is acceptable because the search
runs only on accounts already flagged - hundreds of alerts, not five million
transactions.

**Feature set stays at 49. PR-AUC stays at 0.1522.**

---

# Blend and calibration - and a reversal

Applied the card rail's two proven steps: rank-average LightGBM with XGBoost,
then calibrate on a held-out slice (fit 85% / early stop 7.5% / calibrate 7.5%).

| Model | PR-AUC | Brier | vs baseline | Calibrated? |
| --- | --- | --- | --- | --- |
| LightGBM alone | 0.1035 | 0.001800 | -1.9% | **No** |
| **XGBoost alone** | **0.1831** | **0.001599** | **+9.5%** | **Yes** |
| Blend, raw | 0.1619 | 0.328705 | - | No |
| Blend + Platt | 0.1619 | 0.001709 | +3.2% | Yes |
| Blend + isotonic | 0.1500 | 0.001618 | +8.4% | Yes (133 scores) |

## XGBoost alone wins here, and the blend hurts

This is the reverse of the card rail, where the blend beat both parents on all
three seeds. Here XGBoost alone is **+13% over the blend** and **+77% over
LightGBM**, and it is already well calibrated without any Platt layer.

Blending only helps when two models are wrong in uncorrelated ways. On the card
rail LightGBM and XGBoost were close in strength (0.5865 vs 0.5663), so
averaging cancelled noise. Here they are not close - 0.1831 against 0.1035 - so
averaging drags the stronger model down toward the weaker one. That is not a
surprising result once stated; it is a reminder that a technique proven on one
rail is a hypothesis on the next, not a conclusion.

## LightGBM got worse, and that is the calibration cost

0.1522 in the previous experiment, 0.1035 here. Nothing about the model changed
- it now fits on 85% of the training period instead of 90%, because 7.5% is
held out for calibration. **A third of its PR-AUC for a calibration slice** is a
steep price, and far steeper than the card rail's (0.5948 -> 0.5785, about 3%).

XGBoost absorbed the same reduction and still scored 0.1831, which is further
evidence the two models are genuinely different here rather than
interchangeable.

## Isotonic collapsed the scores again

133 distinct values from 1,015,669 rows - the same step-function failure
measured on the card rail, where it produced 153. Platt gave 817,840 distinct
scores and preserved the ranking exactly (PR-AUC identical to the raw blend to
four decimals). The card rail's conclusion holds on this rail unchanged.

## This is one seed

The card rail measured XGBoost's seed spread at **+-0.0170 against LightGBM's
+-0.0024** - seven times less stable. A single XGBoost run is exactly the
measurement least worth trusting, and the +1.22% V-column result that turned out
to be -0.71% across three seeds is the standing warning.

The gap here is large (+77% over LightGBM), so it is unlikely to be pure noise -
but "unlikely" is not "measured". **Not adopting XGBoost-alone as the wire model
until it is run across three seeds.**

## Current best, pending that check

| Flag rate | Laundering caught | Precision | False alarms per catch |
| --- | --- | --- | --- |
| 0.1% | 14.7% | 26.0% | 2.8 |
| 0.5% | 37.5% | 13.3% | 6.5 |
| **1.0%** | **56.4%** | 10.0% | 9.0 |
| 2.0% | 76.5% | 6.8% | 13.8 |
