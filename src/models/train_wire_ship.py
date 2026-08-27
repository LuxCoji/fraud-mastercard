"""Train and export the shippable wire model: XGBoost on pattern features.

The artifact in `artifacts/wire_lightgbm.txt` expects 25 features - it predates
the pattern-count work that lifted PR-AUC 45%, and the current builder produces
49. Loading it into the router therefore fails outright with a shape mismatch,
which is how the discrepancy surfaced.
"""
import sys, warnings, json; sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb
import config as cfg
from data import loaders
from features.wire import LABEL, WireFeatureBuilder, sort_by_time
from models.metrics import headline, operating_points

df = sort_by_time(loaders.load_wire())
df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
split = int(len(df) * 0.80)
train_df, test_df = df.iloc[:split], df.iloc[split:]
y_train, y_test = train_df[LABEL].values, test_df[LABEL].values
fit_end = int(len(train_df) * 0.9)

print("building features...", flush=True)
builder = WireFeatureBuilder(with_patterns=True).fit(train_df.iloc[:fit_end])
X_fit = builder.transform(train_df.iloc[:fit_end])
X_stop = builder.transform(train_df.iloc[fit_end:])
X_test = builder.transform(test_df)
print(f"  {X_fit.shape[1]} features", flush=True)

params = {"objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 8,
          "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.4,
          "tree_method": "hist", "nthread": -1, "seed": 42}
dtrain = xgb.QuantileDMatrix(X_fit, label=y_train[:fit_end], missing=np.nan)
dvalid = xgb.QuantileDMatrix(X_stop, label=y_train[fit_end:], missing=np.nan, ref=dtrain)
bst = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dvalid, "valid")],
                early_stopping_rounds=100, verbose_eval=250)
scores = bst.predict(xgb.DMatrix(X_test, missing=np.nan),
                     iteration_range=(0, bst.best_iteration + 1))
h = headline(y_test, scores)
pts = operating_points(y_test, scores)
at1 = pts[pts.alert_rate == 0.01].iloc[0]
print(f"\nrounds {bst.best_iteration}  PR-AUC {h['pr_auc']:.4f}  recall@1% {at1.recall:.2%}")

bst.save_model(str(cfg.ARTIFACTS / "wire_xgboost.json"))
(cfg.ARTIFACTS / "wire_feature_names.json").write_text(json.dumps(list(X_fit.columns)))
(cfg.ARTIFACTS / "wire_model_card.json").write_text(json.dumps({
    "rail": "wire", "source": "ibm_aml_hi_small", "model": "xgboost",
    "features": int(X_fit.shape[1]), "with_patterns": True, "seed": 42,
    "rounds": int(bst.best_iteration), "synthetic_source": True,
    "metrics": {"pr_auc": float(h["pr_auc"]), "auc": float(h["auc"]),
                "brier": float(h["brier"]),
                "recall_at_operating_point": float(at1.recall),
                "precision_at_operating_point": float(at1.precision)},
    "note": "HI-Small. HI-Medium measured 68.65% recall@1% but its artifacts "
            "were never exported - see reports/step7_hi_medium.md"}, indent=2))
np.save(cfg.INTERIM / "wire_scores_shipped.npy", scores)
print(f"saved wire_xgboost.json with {X_fit.shape[1]} features")
