# Step 14 - Flow features: rejected

The flow tracer reaches **79% of the labelled rings** against 19% for the
account-graph detectors it replaced, and it had only ever been wired into case
display - the wire model used 49 features and not one came from it. That looked
like the largest available gap.

It was not.

## Result

| Seed | Patterns only | + traced flows |
| --- | --- | --- |
| 42 | 58.38% | 56.32% |
| 7 | 55.48% | 56.32% |
| 2024 | 55.76% | 57.37% |
| **mean** | **56.54%** | **56.67%** |

**Gap +0.2%, and it loses on seed 42.** Against a 1.2% seed spread on this rail,
that is noise. Rejected.

Flow features carry **9.3% of total gain** - the model uses them, it just does
not get more right because of them.

## Why, most likely

**The pattern features already carry the same information.** `graph_patterns.py`
supplies cycle counts, tight-cycle counts, fastest-cycle hours, fan widths and
pass-through ratios. A flow's shape is largely reconstructible from those: an
account with many tight cycles and a high pass-through ratio is an account in
the middle of fast circular flows, which is exactly what `flow_middle_ratio` and
`flow_cycle_ratio` say more directly.

This is the third time the same thing has happened on this project:

| Change | Why it failed |
| --- | --- |
| Chain and bipartite features (step 4c) | -7% |
| Cross-border thresholds (step 11) | the model already priced geography in |
| LLM consensus triage (step 13) | reasoning over evidence the model already had |
| **Flow features (here)** | **the pattern counts already encode the shape** |

The pattern is worth stating as a rule: **on this rail, a new signal derived from
the graph the model already sees is unlikely to add anything.** Adding a fourth
view of the same structure is not adding information.

## What the tracer is still for

**Explaining a case, which is a different job and one it does well.** The paths
it produces are the difference between "score 0.99" and

```
812D22980 -> 811C599A0 -> 811C597B0 -> 811B83280 -> 811C599A0 -> 812D22980
69,597 first hop, 10,748 last, over 76h
```

An investigator can act on the second. Nothing in the feature set produces it,
and its value does not depend on improving PR-AUC. `flows.py` stays wired into
`cases.py` and out of `wire.py`, and `with_flows` defaults to off.

## A note on the run

Three local attempts were OOM-killed before this moved to Kaggle. The flow graph
was unbounded - **41.9 million edges**, because every transaction linked to every
onward payment inside a 120-hour window. Capping onward edges at six per
transaction took it to 4.7 million, and streaming the flow statistics instead of
materialising 400,000 `Flow` objects removed the rest of the pressure.

One baseline seed had also early-stopped at **1 round**, scoring 47.02% from a
single stump. Runs under ten rounds are now excluded from the comparison - a
stump setting the baseline would have made any change look good for free.
