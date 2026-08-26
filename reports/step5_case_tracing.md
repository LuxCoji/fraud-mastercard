# Step 5 - Case tracing: does a flagged score become an investigable case?

The wire rail produces 10,156 alerts at its 1% operating point. Alerts are not
work items - a ring producing twelve alerts is one operation, not twelve. This
measures whether grouping and tracing them produces something an investigator
can actually act on.

## The queue

| | |
| --- | --- |
| Alerts at 1% | 10,156 |
| **Cases** | **3,895** |
| Alerts per case | 2.6 |
| Transaction precision | 10.0% |
| Case precision | 9.3% |
| Labelled rings reached | **319 of 370 (86.2%)** |

**The queue is 62% shorter** - 3,895 units of work instead of 10,156 - for
essentially the same precision. That alone is the case for grouping.

## Ranking by case is far better at the top

**Of the top 20 cases, 16 contain real laundering - 80%.**

Against a 9.3% average across all cases, and a 0.18% base rate in the underlying
data. An investigator working the top of this queue finds something four times
in five.

The mechanism is simple: a case's score is the maximum of its alerts, so a case
containing one very high-scoring transaction ranks high, and the surrounding
alerts come along as context. Ranking transactions individually scatters an
operation's evidence across the queue.

## Tracing works, but reaches only a minority of cases

| | Share |
| --- | --- |
| Cases with a traced structure | 214 (5.5%) |
| **Real cases with a traced structure** | **60 (16.6%)** |

So five cases in six arrive with **"no structure found"**. The score says
suspicious; the trace cannot say why.

That is a genuine limitation and it follows directly from the chain detector
already measured as weak - it found 7 of 43 STACK rings and 4 of 49 BIPARTITE.
Cycles trace well; nothing else does.

## A bug this run caught

The first pass reported most top cases as `CYCLE (1 hops)`. A one-node cycle is
an account paying **itself** - which IBM AML's `Reinvestment` payment format does
routinely and legitimately. The detector was reporting ordinary reinvestment as
laundering evidence.

Excluding self-loops dropped cases-with-structure from 15.4% to 5.5% and
real-cases-with-structure from 51.4% to 16.6%. **Two thirds of the apparent
tracing success was self-loops.** The lower numbers are the true ones.

## A worked case

```
CASE 0
  alerts        26
  accounts      4
  total value   15,209,516
  highest score 0.9930
  evidence      CYCLE (2 hops)

  cycles (15):
    811C599A0 -> 811B83280 -> 811C599A0
    811C599A0 -> 811C597B0 -> 811C599A0
    811C599A0 -> 812D22980 -> 811C599A0
    811B83280 -> 812D22980 -> 811B83280
```

Four accounts, fifteen distinct loops between them, 15.2 million moved. That is
a complete, self-contained thing to hand to a reviewer - not a score.

## What this means for the freeze workflow

The 80% precision at the top of the queue is what makes a review process
workable: a reviewer opening the top twenty cases is not wading through noise.

It does **not** justify automatic freezing. Case precision across the whole
queue is 9.3%, so freezing on the model's say-so would freeze roughly ten
innocent parties per real one. The design stands: **flag, trace, human approves,
then freeze.**

## Next on this rail

- **Structure coverage is the weak point**, not ranking. Five cases in six have
  no explanation attached, and that is the chain detector's known weakness
  rather than a scoring problem.
- Fan-in and fan-out are counted as *features* but never *traced*, even though
  they are the easiest shapes to render. Adding them to the trace would lift
  coverage without new algorithms.
