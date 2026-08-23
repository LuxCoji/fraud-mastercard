# fraud_mastercard

Payment-fraud detection across three rails, built to minimise **false positives
at a fixed review budget** rather than to maximise leaderboard AUC.

## Design in one paragraph

A transaction arrives on a known rail. Routing between rails is deterministic
from the request payload, never learned. Each rail gets its own feature space,
its own gradient-boosted model, and its own calibrated threshold; they share one
entity/velocity feature store and one policy layer. Only one model runs per
transaction, so adding rails costs no latency.

```
request
  |
  +--> rail router            deterministic, from payload       ~0 ms
  +--> shared entity features near-cache, rail-agnostic         <1 ms
  +--> rail-specific head     ONE head runs, not three          1-3 ms
  +--> policy engine          per-rail thresholds               ~0 ms
         allow / step-up / block
```

## Rails

| Rail | Source | Rows | Base rate | Model |
| --- | --- | --- | --- | --- |
| `card` | IEEE-CIS | 590,540 | 3.499% | LightGBM (CatBoost challenger) |
| `transfer` | PaySim | 6,362,620 | 0.129% | LightGBM |
| `wire` | IBM AML HI-Small | 5,078,345 | 0.102% | LightGBM on graph features |

## Rules this project is built on

These are conclusions carried over from a previous attempt, each backed by a
measurement rather than a preference.

1. **No canonical cross-rail schema.** The previous attempt mapped all three
   sources into shared feature slots and dropped IEEE-CIS's `V1`-`V339` because
   their semantics were unrecoverable. That is where the signal lives; it scored
   0.073 PR-AUC as a result. Each rail keeps its own columns.
2. **No class rebalancing. Ever.** No `class_weight="balanced"`, no SMOTE.
   Measured previously: Brier 0.2025 -> 0.0039 on removal, while PR-AUC barely
   moved. Ranking metrics hide calibration damage, and every policy threshold is
   derived from a probability.
3. **Never pool rails in a metric.** Base rates span 34x. A pooled figure
   measures the sampling ratio, not skill.
4. **Chronological, entity-grouped validation.** Random CV leaks the same card
   into train and test and inflates everything downstream.
5. **The autoencoder is a feature, not a scorer.** It lost to plain logistic
   regression on 2 of 3 sources alone, but ranks 3rd of 21 features inside a
   gradient-boosted fusion (+109% PR-AUC on IBM AML). It earns its 3 ms only
   where per-user history exists.
6. **Optimise alert rate at target recall**, not AUC.

## Layout

```
data/raw/          source CSVs (gitignored)
data/interim/      Parquet caches (gitignored)
src/config.py      all paths and rail constants
src/data/          per-rail loaders + verification gate
src/features/      per-rail feature builders
src/models/        training and evaluation
src/validation/    chronological + entity-grouped splits
src/policy/        calibration and thresholds
reports/           per-stage measured results
```

## Reproducing

```bash
pip install -r requirements.txt
python src/data/verify.py        # Step 1 gate: proves all three sources load
```

## Status

- [x] Step 0 - old Cloud Run deployment scaled to zero
- [x] Step 1 - data staged and verified
- [ ] Step 2 - card head
- [ ] Step 3 - transfer head
- [ ] Step 4 - wire head
- [ ] Step 5 - router and policy
