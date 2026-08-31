# Red-team gaps against what is actually built

Re-checked against the `risk/` package on `feat/fraud-detection`. **The earlier
version of this file claimed detectors that do not exist**, so the verdicts have
moved. Two gaps it recorded as "catches" were misses, and both are now fixed.

The wire rail has exactly five signals, all hand-written and none trained:
`passthrough`, `fan_out`, `fan_in`, `send_burst`, `tight_cycle`. There is no
bipartite detector, no scatter-gather detector, and no ring tracing.

| Gap | Verdict | Why |
| --- | --- | --- |
| 1. Unlimited inbound volume | **Catches** | `fan_in` fires at 8 distinct payers inside six hours |
| 2. Destination status ignored | **No** | Not a model job - a missing check in the payment engine |
| 3. Same owner, three accounts | **No** | The graph keys on accounts. `RiskContext` carries `actor_id`, but the rail never links accounts to an owner |
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

So the ceilings are split: **0.75 for a step-up, 0.97 for a block.** A confident
model can now challenge a card test, while a hard decline on a small payment
still asks for near-certainty - a step-up is recoverable in ten seconds, a
decline is a lost sale.

## Still open

**Gaps 2 and 3 are not model problems.** Checking the destination account's
status before processing is an engine rule. Linking accounts by owner needs the
wire graph to key on owner as well as account - `RiskContext` already carries
`actor_id`, so the data is there and the graph simply does not use it.

**Gaps 7 and 8 need a detector that does not exist.** Long-chain layering and
N-to-M routing that never closes a cycle are both invisible to a bounded cycle
search. A pass-through propagation walk - following money that arrives and
leaves quickly, hop by hop - would find both, and is the obvious next detector.

**None of this is measured against the red team.** Every verdict above is "does
a detector exist and would it fire on this shape", which is a reading of the
code, not a result. The number that matters comes from running the LLM agent
against the rails and counting what it gets past them.
