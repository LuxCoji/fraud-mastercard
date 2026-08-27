# Step 12 - Flow tracing: 19% to 79% ring coverage

Five cases in six arrived at the investigator saying "no structure found". The
score said suspicious; the trace could not say why.

## The cause was the graph, not the algorithm

The account graph collapses every payment between A and B into one edge with a
summed total and a first/last timestamp. That is right for "how many
counterparties does this account have" and useless for "did this money move
onward" - **the sequence is exactly what got summed away**.

So the chain detector built on it reached 7 of 43 STACK rings and 4 of 49
BIPARTITE, and accidentally found cycles better than the chains it was written
for.

## The fix: transactions as nodes

A node is one transaction. An edge means *this money could have funded that
payment*:

```
txn_i (A -> B at t_i)  ->  txn_j (B -> C at t_j)

  when  t_j > t_i          the onward payment follows the arrival
  and   t_j - t_i <= 120h  it moved on promptly
  and   0.1 <= amount_j / amount_i <= 10   it carries a comparable sum
```

Two properties follow, and both matter:

**The graph is acyclic.** Time strictly increases along every edge, so traversal
cannot loop. That is why this costs seconds where account-graph cycle
enumeration exhausted a 900-second budget on HI-Medium.

**A laundering cycle is a path, not a cycle.** A -> B -> C -> A is three
transactions in time order whose last receiver is the first sender, so one
traversal finds layering chains and circular flows together and tells them apart
by comparing endpoints.

## Ring coverage, measured against the 370 labelled rings

| Pattern | Rings | **Flow graph** | Account graph |
| --- | --- | --- | --- |
| CYCLE | 54 | **54 (100%)** | 25 (46%) |
| GATHER-SCATTER | 51 | **51 (100%)** | 5 (10%) |
| SCATTER-GATHER | 44 | **43 (98%)** | 1 (2%) |
| STACK | 43 | **42 (98%)** | 7 (16%) |
| RANDOM | 41 | **39 (95%)** | 19 (46%) |
| BIPARTITE | 49 | **25 (51%)** | 4 (8%) |
| FAN-OUT | 48 | **20 (42%)** | 4 (8%) |
| FAN-IN | 40 | **17 (42%)** | 4 (10%) |
| **TOTAL** | **370** | **291 (79%)** | **69 (19%)** |

**STACK went from 16% to 98%** - the shape the account-graph detector was
written for and could not find. Fan patterns remain lowest, which is expected: a
fan is one hop wide, so there is little path to trace, and the existing fan
detector already covers them.

## A bug worth recording

The first version produced 50,000 flows in 1.3 seconds and reached **fewer**
rings than the account graph it replaced. Every flow was a variation of the same
path: the traversal ping-ponged between a pair of accounts, filling the entire
budget while touching a handful of accounts.

Capping each account at **two appearances per flow** fixed it - once outbound
and once returning, which is what closes a cycle. Coverage went to 79%, and 200
flows now come from 1,788 distinct accounts rather than a dozen.

Starting from transactions nothing feeds into helps too: beginning mid-path only
re-discovers suffixes of paths the sources already produce.

## What an investigator now opens

```
CASE 0
  alerts        26      accounts 4      total value 15,209,516
  evidence      CYCLE (5 hops, 76h)

  traced flows (200):
    CYCLE 5 hops | 812D22980 -> 811C599A0 -> 811C597B0 -> 811B83280
                   -> 811C599A0 -> 812D22980
                 | 69,597 in / 10,748 out (15%) over 76h
```

Four accounts, money leaving and returning across five transactions in 76 hours,
15% retained. That is a case a human can act on.

## Effect on the queue

| | Before | After |
| --- | --- | --- |
| Cases with a traced structure | 21.5% | **22.0%** |
| Real cases with a structure | 31.5% | **35.9%** |
| Top-20 precision | 80% | 80% (unchanged) |

The queue-level numbers move less than the ring coverage does, and the reason is
worth being clear about: a case is built from the alerts inside it, so if a ring
contributed only one alert there is no path to trace within that case. **Ring
coverage measures what the tracer can find; case coverage measures how much of a
ring the alert queue actually surfaced.** Improving the second means catching
more transactions per ring, which is a scoring problem rather than a tracing one.

Ranking is untouched - flows explain, they never re-score.
