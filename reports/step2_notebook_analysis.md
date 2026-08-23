# Step 2a - What the reference notebooks actually do

Four kernels pulled to `notebooks/reference/`. Everything below is read from
their source, not from recollection.

## Models and hyperparameters, as published

**Chris Deotte - XGB Fraud with Magic [0.9600]**

```python
XGBClassifier(
    n_estimators=2000,        # 5000 in the second variant
    max_depth=12,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.4,
    missing=-1,
    eval_metric='auc',
    tree_method='gpu_hist',   # we have no GPU -> 'hist'
)
```

**Konstantin Yakovlev - LGBM with GroupKFold CV**

```python
lgb_params = {
    'objective': 'binary',  'boosting_type': 'gbdt',  'metric': 'auc',
    'learning_rate': 0.01,  'num_leaves': 2**8,       'max_depth': -1,
    'colsample_bytree': 0.5,                          # 0.7 in his FE kernel
    'subsample': 0.7,       'subsample_freq': 1,
    'n_estimators': 800,                              # 80000 in the FE kernel
    'max_bin': 255,         'early_stopping_rounds': 100,
    'tree_learner': 'serial',
}
```

No `reg_alpha` or `reg_lambda` in either dict - regularisation is carried by
`colsample_bytree` and `subsample` alone.

## Three corrections to what I previously told you

**1. They do not rebalance. Nothing to remove.**
No `scale_pos_weight`, `is_unbalance`, `class_weight` or SMOTE appears in any of
the four notebooks. My instruction to "remove the rebalancing setting" was aimed
at the wrong target: `class_weight="balanced"` was a defect in the *previous
project*, not in these kernels. Their params are calibration-safe as published.

**2. Their CV is better than I said.** Both use `GroupKFold(n_splits=6)` grouped
on `DT_M` - the calendar month - holding out one whole month per fold. That is
time-aware, not the random K-fold I described.

**3. But it is still not entity-grouped**, so the original concern survives in a
weaker form. Grouping by month stops a transaction leaking into its own fold; it
does not stop the same *card* appearing in month 1 (train) and month 5 (holdout).
Cards recur across months constantly. Entity leakage persists - it is simply
smaller than under random K-fold.

## The "magic", and how much of it is legal in production

`uid = card1_addr1_D1n` (card1 + addr1 + a D1 normalised to a registration
date). Deotte believes each UID is one physical credit card.

Two distinct mechanisms, and they transfer very differently:

| Mechanism | What it is | Transfers to production? |
| --- | --- | --- |
| **UID aggregations** | Group stats (`mean`, `std`, `nunique`) of `TransactionAmt`, `D9`, `D11` per UID, fed as features. The UID itself is never a model input. | **Partly.** Legal over a UID's *past*. Their version aggregates over train+test jointly. |
| **UID post-processing** | Every transaction of one UID is *all* fraud or *all* legitimate, so final predictions are overwritten with the UID's group mean. | **No.** Requires seeing the UID's future transactions. |

The second is where much of the 0.9600 comes from, and it is transductive by
construction: the Kaggle test set is handed to you in full, so a UID's later
transactions are visible when scoring its earlier ones. A live authorisation has
no such future. We can use a UID's history; we cannot use its future.

## Confirmed leakage pattern

```python
temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
temp_df = temp_df.groupby([col])[main_column].agg([agg_type])
```

Every aggregation and frequency encoding is fitted on train **and** test
together. Standard and legitimate for a Kaggle transductive setting; invalid for
us. All encoders must be fitted on train only and applied forward.

## Also worth taking

- 219 of the 339 V columns are dropped as redundant by correlation analysis
  (`cdeotte/eda-for-columns-v-and-id`). ~120 survive. Cheaper and no worse.
- `cents = TransactionAmt - floor(TransactionAmt)` - the decimal part alone is
  predictive.
- Frequency encoding on `addr1`, `card1`, `card2`, `card3`, `P_emaildomain`.
- Combination features `card1_addr1`, `card1_addr1_P_emaildomain`.

## What we keep, change, and drop

| | Decision |
| --- | --- |
| Model family | **Keep** - LightGBM. XGBoost scores the same; LGBM fits 6 GB RAM. |
| Hyperparameters | **Keep as published.** Re-fit `n_estimators` by early stopping on our split. |
| `tree_method` | **Change** - `gpu_hist` -> `hist`. No GPU here. |
| Feature engineering | **Keep** - UID aggregations, frequency/combination encoding, cents. |
| Encoder fitting | **Change** - train-only, applied forward. No train+test concat. |
| CV grouping | **Change** - month **and** entity, not month alone. |
| UID post-processing | **Drop** - needs the UID's future. |
| Metric | **Change** - AUC -> alert rate at target recall. |

## Expected consequence

Our AUC will land **below** the 0.9600 headline. Removing the transductive
post-processing and the joint-fitted encoders removes real predictive power that
genuinely does not exist at authorisation time. A lower honest number is the
correct outcome, and the gap between the two is the measurement worth reporting.
