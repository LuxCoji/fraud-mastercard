# Step 8 - Router and API

Two models existed and nothing joined them. This is the join.

## Routing is deterministic

A transaction arrives on a known rail and the payload says which - a card
authorisation carries `card1`, a wire carries `From Bank`. Routing reads that.
No model, no inference, no failure mode.

## The rails are deliberately asymmetric

| | card | wire |
| --- | --- | --- |
| Base rate | 3.44% | 0.17% |
| Precision at operating point | ~80% | ~10% |
| Runs | inside the 100 ms window | asynchronously |
| Decisions | allow / step up / **block** | allow / **investigate** |

**The card rail decides. The wire rail recommends.** At 10% precision an
automatic freeze would stop roughly nine innocent parties per real one, so the
wire rail queues a case for a human and never blocks. That is encoded in the
router rather than left to whoever calls it.

The transfer rail routes but returns `scored=False`. PaySim's fraud is separable
by a three-line rule that holds for 97.8% of fraud and 0.00% of legitimate
traffic - the simulator showing through - so there is no trustworthy model to
call. Allowing unscored is honest; a fabricated score would be worse than none.

## Measured on 5,000 held-out transactions

| Band | Share | Precision | Share of all fraud |
| --- | --- | --- | --- |
| **block** | 0.68% | **85.3%** | 12.7% |
| **step up** | 1.72% | **80.2%** | 30.3% |
| allow | 97.60% | 2.7% | 57.0% |

**Genuine customers blocked: 0.105% - about 1 in 954.**

Latency p50 5.67 ms, p99 15.9 ms, against a 100 ms budget.

## A bug worth recording

The first run put **1.8% of traffic in the block band where 0.5% was intended**,
and squeezed step-up to 0.36% where 1.5% was intended.

Thresholds had been loaded from `card_scores_final.npy` - the **Platt-calibrated
rank average** - while the router returns a **raw mean of the two models**. Two
different score distributions, so cut-offs fitted on one land in the wrong place
on the other.

The ordering was right throughout (block 89% precision, step-up 61%, allow
2.8%), which is exactly what makes this kind of bug easy to miss: nothing looks
broken, the bands are just the wrong size, and the business consequence is four
times as many hard declines as intended.

**Thresholds are now fitted on scores from the same function the router calls.**

## Thresholds are set by rate, not by score

`thresholds_from_scores` converts "block the top 0.5%" into a score cut-off
against held-out data. A business can reason about "we decline one transaction
in two hundred"; nobody can reason about 0.8620, and that number changes
whenever the model is retrained.

## API

`src/api.py` - `POST /score` with a transaction, get a decision; `GET /health`
reports loaded state and current thresholds. Deliberately thin: everything that
decides anything lives in `router.py`, so the policy is testable without a
server and a caller could embed the scorer directly.

Models load once at startup. Loading a LightGBM booster costs tens of
milliseconds - fine once, fatal per request.
