# Step 15 - Support-agent context, and what it cannot say

When a customer calls about a declined card payment, the agent sees a score of
0.94 and nothing else. This turns that into something readable.

It is deliberately **not** an explanation of the model's decision, and the
measurement below is why.

## Only a quarter of the model's reasoning is explainable

Measured on the 200 highest-scoring card transactions, over the evidence that
pushed each score **up**:

| | |
| --- | --- |
| Explainable share, mean | **25.3%** |
| Explainable share, median | **18.7%** |
| Cases under a quarter explainable | **82%** |
| Cases over half explainable | 13% |

The rest sits on `V258`, `C1`, `V187` and their neighbours - columns IEEE-CIS
anonymised deliberately, whose meaning cannot be recovered.

An explanation built on a quarter of the reasoning would sound authoritative and
be mostly wrong, and an agent repeating it would tell a customer something
false. **That is worse than saying nothing**, and it is a different thing legally
from telling a customer their payment looked unusual.

## So it answers a different question

Not "why did we decline this" but **"what is factually unusual about this
transaction, compared with what this card normally does"**. Every line is a
measured comparison against that card's own history.

```
- Amount: 27
- Against this card's usual spend: typically 40 - broadly in line
- History for this card: 4 previous transactions on record
- Time of day: 23:00
- Billing country: not supplied with this payment
```

becomes

> The payment was for $27, which is close to the card's usual $40 spend and there
> have only been four prior transactions on this card. It was made at 23:00 and
> the billing country was not supplied, which is the only detail that stands out.

An agent can read that aloud. It claims nothing it cannot support.

## The finding an operator needs to know

**Of the 40 highest-scoring blocks, only 18 have anything readable that looks
unusual at all.** The other 22 were blocked on evidence with no human meaning,
and the honest summary for those says so.

That is not a defect in this module - it is the truth about the model, surfaced.
A support process built on the assumption that every decline can be explained
would break on 55% of calls; one built on this knows in advance which calls have
an answer.

## Grounding is checked, not trusted

Every number in the summary is compared against the numbers in the facts. A
figure the model introduced on its own fails the check and is reported. Three
tests cover it, including a case where a plausible invented figure is caught.

## Not wired in

`explain.py` is standalone. Wiring it into the console means calling an LLM per
declined transaction, and the measurement above says that is worth doing for
roughly 45% of them. Which of the two - always call, or call only when the facts
show something - is a cost decision rather than a technical one.
