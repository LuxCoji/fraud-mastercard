Short answer: **5 of 8 clearly, 2 partly, 1 not at all.**

| Gap | Our system | Why |
|---|---|---|
| 1. Unlimited inbound volume | **Catches** | Wire rail has fan-in as a trained pattern — money pouring into one account from many is its bread and butter |
| 2. Destination status ignored | **No** | This isn't a model job. It's a missing check in the payment engine |
| 3. Same owner, 3 accounts | **Partly** | It sees "3 sources → 1 target" as a fan-in shape. It does **not** know the 3 accounts share an owner |
| 4. Midnight counter reset | **Partly** | Our model isn't a daily counter, so resetting one doesn't blind it — but it doesn't specifically flag the trick either |
| 5. 15 rapid micro-transfers | **Catches** | Time-since-last-transaction and count features are core to both rails; TREASURE reads exactly this kind of burst |
| 6. Micro fan-out below limits | **Catches** | Fan-out and scatter-gather are named laundering shapes in the wire rail's training data |
| 7. Multi-hop layering, 16 steps | **Catches — best match** | Ring tracing follows the chain across hops and shows the whole path in the console |
| 8. N-to-M cyclic routing | **Catches** | Cycle and bipartite are both trained patterns; bipartite *is* N sources → M destinations |

**Three honest caveats:**

**This is "has the right detector", not "measured on your red team".** The wire rail learned these shapes from IBM's laundering dataset, where they're labelled. FinSim traffic is different. Nothing has been measured against these four sessions yet — that measurement is the point of wiring it in.

**Gaps 2 and 4 aren't model problems.** They're two missing rules in the engine: check the destination account's status before processing, and use a rolling window instead of a calendar day. Both are small, cheap fixes — worth doing regardless of the model.

**The amounts here are small, and that cuts against us.** ₹95,000–₹500,000. Our thresholds are dynamic by value, so a small transfer needs a *higher* fraud score to trip than a large one. The red team's structuring is deliberately low-value — that's the whole idea of structuring. Expect the wire rail to catch the shape but sit lower in the queue than the number suggests it should. Worth revisiting the amount curve once we have real numbers.