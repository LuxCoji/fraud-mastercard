"""HTTP interface: POST a transaction, get a decision.

Deliberately thin. Everything that decides anything lives in `router.py`; this
file is transport. That keeps the policy testable without a server and means a
caller could bypass HTTP entirely if they wanted to embed the scorer.

Models load once at startup, not per request. Loading a LightGBM booster costs
tens of milliseconds - fine once, fatal inside a 100 ms authorisation.

Run:
    uvicorn src.api:app --port 8000
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as cfg
from router import CARD_ALERT_RATE, CARD_BLOCK_RATE, FraudRouter, thresholds_from_scores

_state: dict[str, Any] = {"router": None, "loaded_at": None, "thresholds": {}}


def build_router() -> FraudRouter:
    """Load models and fit thresholds. Called once."""
    import lightgbm as lgb
    import xgboost as xgb

    from data import loaders
    from features.card import CardFeatureBuilder
    from models.scorer import FastCardScorer
    from validation.splits import chronological_split

    df = loaders.load_card()
    split = chronological_split(df, test_fraction=0.20)
    builder = CardFeatureBuilder().fit(df.iloc[split.train_idx])

    X_train = pd.read_parquet(cfg.INTERIM / "card_X_train.parquet")
    y_train = np.load(cfg.INTERIM / "card_y_train.npy")
    feature_names = list(X_train.columns)

    lgb_model = lgb.Booster(model_file=str(cfg.ARTIFACTS / "card_lightgbm.txt"))
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.4, missing=np.nan, eval_metric="aucpr",
        tree_method="hist", n_jobs=1, random_state=cfg.RANDOM_SEED)
    xgb_model.fit(X_train, y_train)

    card_scorer = FastCardScorer(builder, feature_names, [lgb_model, xgb_model])

    # Thresholds must be fitted on scores from *this* scorer, not from a saved
    # array. The stored scores are Platt-calibrated rank averages; the router
    # returns a raw mean of the two models. Mixing them put 1.8% of traffic in
    # the block band where 0.5% was intended, and squeezed step-up to 0.36%
    # where 1.5% was intended - the ordering was right and the sizes were wrong.
    #
    # Held-out rows only: fitting on training scores would place the cut-offs
    # where the model is overconfident.
    X_test = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")
    held_out = np.mean([lgb_model.predict(X_test),
                        xgb_model.predict_proba(X_test)[:, 1]], axis=0)
    block_t, alert_t = thresholds_from_scores(held_out)
    _state["thresholds"] = {
        "card_block": block_t, "card_alert": alert_t,
        "card_block_rate": CARD_BLOCK_RATE, "card_alert_rate": CARD_ALERT_RATE,
    }

    return FraudRouter(card_scorer=card_scorer,
                       card_block_threshold=block_t,
                       card_alert_threshold=alert_t)


def get_router() -> FraudRouter:
    if _state["router"] is None:
        t0 = time.perf_counter()
        _state["router"] = build_router()
        _state["loaded_at"] = time.time()
        print(f"models loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return _state["router"]


# --- FastAPI, imported lazily so the module is usable without it -------------
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="fraud router", version="1.0")

    class ScoreRequest(BaseModel):
        transaction: dict

    @app.on_event("startup")
    def _startup() -> None:
        get_router()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "models_loaded": _state["router"] is not None,
                "thresholds": _state["thresholds"]}

    @app.post("/score")
    def score(request: ScoreRequest) -> dict:
        """Score one transaction and return the decision.

        `latency_ms` is the server-side cost only - it excludes network time,
        which is the caller's to measure.
        """
        t0 = time.perf_counter()
        decision = get_router().decide(request.transaction)
        return {**decision.to_dict(),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

except ImportError:  # FastAPI is optional; the router works without it
    app = None
