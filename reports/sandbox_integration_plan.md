# Integrating the fraud system into sandbox_payments (FinSim)

## What is already there

FinSim is a deterministic, event-sourced payment simulator with branching
("ChronoDAG") and an LLM **red-team agent** that probes it for fraud.

**There is no blue team.** `agents/` contains `redteam/` and nothing else, and a
search for `risk_score`, `fraud_score` or any scoring function returns nothing.

The red-team design assumes a detector exists. Its own diagram
(`docs/redteam_agent_design.md`) reads:

```
├─ fork -> attempt-a   agent tries: rapid cash-out
│                      -> flagged, rolled back
├─ fork -> attempt-b   agent tries: structured transfers under threshold
│                      -> clears undetected
```

Nothing does the flagging. The doc also states the agent's "entire job is to
probe and react to detection". **We are the missing half.**

## Where the hook goes

`sim/core/engine.py` decides every payment and transfer:

| Line | Method | Currently checks |
| --- | --- | --- |
| 470 | `_execute_transfer` | ownership, balance |
| 535 | `_execute_payment` | ownership, balance |

Both then emit `PaymentAuthorized` or `PaymentDeclined` / `TransferRejected`.
A risk check slots in beside the balance check, in the same place a real
authorisation system puts it.

Three things already exist that we would otherwise have to build:

- **`AccountFrozen`** is already a domain event (`sim/core/events.py:71`), so
  the freeze workflow in `policy/actions.py` has somewhere real to land.
- **`sim/gateway/policy.py`** already holds a rate limiter - a natural sibling
  for a risk policy, and proof the codebase expects checks at this layer.
- **Branching** means a with-detection and without-detection run can be
  compared directly, which is how the detector gets measured rather than
  demonstrated.

## What a Command carries

```python
command_id, actor_id, action_type, source_account_id, target_account_id,
amount_paise, idempotency_key, device_id, gateway_hint, metadata
```

Plus everything the engine knows: account balances, KYC level, full event
history, the merchant, the device.

## The constraint, stated plainly

| Rail | Needs | FinSim has it? |
| --- | --- | --- |
| **Wire** (laundering) | amount, sender, receiver, timestamp | **yes, all of it** |
| **Card** | 292 IEEE-CIS columns incl. `V258`, `C1`, `V187` | **no - 211 of them do not exist** |

Measured: feeding the card model only what FinSim can supply drops it from
**45.3% to 14.1%** caught. It does not error - LightGBM sends a missing value
down the tree's "missing" branch and returns a confident-looking score built on
a third of the evidence. **Nothing warns you.**

The wire model has no such problem. Its 49 features are computed *from the
transaction stream itself*, which FinSim produces.

## Plan

### Stage 1 - Wire rail live (no retraining, works today)

1. Make `fraud_mastercard` installable (`pyproject.toml`), add it as a
   dependency of `sandbox_payments`.
2. `sim/risk/detector.py` - wraps `FastWireScorer`. Maintains the account
   statistics incrementally as events arrive, so scoring stays inside the
   authorisation window.
3. Hook `_execute_transfer` and `_execute_payment`: score, then allow /
   step-up / decline via the existing events.
4. **Determinism check.** FinSim guarantees identical seeds produce identical
   state hashes. The scorer must be deterministic or replay breaks - this is a
   test, not an assumption.

### Stage 2 - Cases and freeze

5. Alerts accumulate into cases via `policy/cases.py`; `AccountFrozen` is
   emitted only after a human approves, exactly as `policy/actions.py` requires.
6. Point the existing console at the sim's live data.

### Stage 3 - Red team vs blue team

7. Run the red agent against a branch with detection on, and the same seed with
   it off. Diff the branches.
8. Report what the agent got away with. **That number is the deliverable** -
   it measures the detector against an adversary that was never told how it
   works.

### The card rail - a decision, not a step

Three honest options, none free:

| | What you get |
| --- | --- |
| **Leave it out of the sim** | Wire rail live and honest; card keeps its 45.3% benchmark, measured on real data, unused in simulation |
| **Replay IEEE-CIS rows through the sim** | Card decisions visible at full accuracy, but those transactions are a recording rather than something the sim generated |
| **Train a FinSim-native card model** | Genuinely live, on device / velocity / account history - but it is a different model and 45.3% does not transfer to it |

## Risks

- **Determinism.** The largest one. If scoring introduces any nondeterminism,
  FinSim's core guarantee is gone.
- **Latency.** The wire scorer is 10.6ms p99. Per-event in a discrete-event
  loop that may matter; incremental statistics are the mitigation.
- **Circularity.** If the red agent and the detector are tuned against each
  other, the resulting number measures the pair, not the detector. The agent
  must stay blind to the model's features.
- **Postgres.** The red-team harness needs `FINSIM_DB_URL`; the in-memory demo
  path does not. Stage 3 needs the database, stages 1-2 do not.
