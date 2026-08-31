# Red-team gaps against what is actually built

> **This file describes the `risk/` package in the `sandbox_payments`
> repository, not this one.** The two systems are different builds and the
> statements below are false about `fraud_mastercard`, which has a
> scatter-gather detector (`src/features/graph_patterns.py`), loads an
> IBM-trained wire model (`artifacts/wire_xgboost.json`, via `src/api.py`), and
> routes transfers to their own rail with their own model
> (`src/router.py`, `artifacts/transfer_lightgbm.txt`). Read it as a note about
> the other repository.

Re-checked against the `risk/` package on `feat/fraud-detection`. **The earlier
version of this file claimed detectors that do not exist**, so the verdicts have
moved. Two gaps it recorded as "catches" were misses, and both are now fixed.

The wire rail has exactly five signals, all hand-written and none trained:
`passthrough`, `fan_out`, `fan_in`, `send_burst`, `tight_cycle`. There is no
bipartite detector, no scatter-gather detector, and no ring tracing.

| Gap | Verdict | Why |
| --- | --- | --- |
| 1. Unlimited inbound volume | **Catches** | `fan_in` fires at 8 distinct payers inside six hours |
| 2. Destination status ignored | **Fixed** | `RiskContext` now carries `destination_status`; a transfer into a frozen, closed or disputed account raises a case on its own |
| 3. Same owner, three accounts | **Fixed** | Both owner ids now travel on the context. Weighted low - moving money between your own accounts is ordinary, and only means something combined with a cycle or a pass-through ratio |
| 4. Midnight counter reset | **Catches** | Every window here is rolling - six hours, a day, a week. There is no calendar-day counter to reset |
| 5. 15 rapid micro-transfers | **Fixed** | Was a miss. Transfers go to the wire rail, which counted only *distinct* partners - fifteen sends across two accounts scored 2. `send_burst` now counts sends |
| 6. Micro fan-out below limits | **Catches** | `fan_out` counts partners regardless of amount; the wire rail never reads the amount at all |
| 7. Multi-hop layering, 16 steps | **Partly** | Cycle search stops at `MAX_CYCLE_LENGTH = 6`. A sixteen-hop chain is found only if a sub-cycle closes inside six hops and one day |
| 8. N-to-M cyclic routing | **Partly** | Caught if the routing closes a tight cycle. There is no bipartite detector, so an N-to-M structure that never closes is missed |

## What the earlier version got wrong

**"The wire rail learned these shapes from IBM's laundering dataset."** It did
not. The rail shipped here is structural: named conditions with hand-set weights.
The IBM-trained model lives in the other repository and was never ported.

**"Ring tracing follows the chain and shows the whole path in the console."**
There is no ring tracing. The graph does a bounded cycle search and the console
prints the resulting reason string.

**"Bipartite is a trained pattern."** There is no bipartite detector.

**"TREASURE reads exactly this kind of burst."** The card model only ever sees
`PAYMENT`. Micro-*transfers* are `TRANSFER`, which never reach it.

**The amount caveat pointed the wrong way.** It warned that dynamic-by-value
thresholds would push low-value structuring down the queue. The wire rail does
not read the amount at all, so structuring is unaffected. The amount curve is
card-only - and there it was a real problem, in the opposite direction from the
one described.

## The amount curve was blocking card testing by construction

Card testing validates a stolen card with tiny charges before using it. Under a
single 0.99 ceiling:

| amount | step-up bar | block bar |
| --- | --- | --- |
| ₹25 | 0.952 | 0.990 |
| ₹1,000 | 0.333 | 0.533 |
| ₹10,000 | 0.050 | 0.080 |

A ₹25 payment needed 0.952 confidence to be challenged - and card testing uses
those amounts *because* a ₹25 loss is not worth stopping. Elkan's rule is right
per transaction and wrong here: the cost of missing it is not ₹25, it is the
large fraud the validated card then funds. A step-up is also not priced by the
transaction - it costs a genuine customer about ten seconds whatever the amount.

So the ceilings are split **in the sandbox**: 0.75 for a step-up, 0.97 for a
block. This repository still uses a single 0.99 ceiling
(`src/policy/cost_threshold.py`) and a fixed percentile for its block bar, and
porting the split here would need `calibrate_fp_cost` re-run - it was fitted
under 0.99, so lowering the step-up ceiling alone would push the alert rate past
the budget the router is sized for.

In the sandbox a confident model can now challenge a card test, while a hard
decline on a small payment still asks for near-certainty - a step-up is
recoverable in ten seconds, a decline is a lost sale.

## Still open

**Gaps 7 and 8 need a detector that does not exist.** Long-chain layering and
N-to-M routing that never closes a cycle are both invisible to a bounded cycle
search. A pass-through propagation walk - following money that arrives and
leaves quickly, hop by hop - would find both, and is the obvious next detector.

**None of this is measured against the red team.** Every verdict above is "does
a detector exist and would it fire on this shape", which is a reading of the
code, not a result. The number that matters comes from running the LLM agent
against the rails and counting what it gets past them.
