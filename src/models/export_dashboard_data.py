"""Export real case and system data for the operations dashboard.

The dashboard reads a single JSON file rather than talking to the models
directly. That keeps it a static page - servable from a bucket, no backend - and
means the expensive work (building the account graph, tracing flows) happens once
here rather than on every page load.

Run: python src/models/export_dashboard_data.py
"""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data import loaders
from features.graph_patterns import build_temporal_graph
from features.wire import LABEL, sort_by_time
from monitoring.drift import Baseline, check
from policy.cases import build_cases

TOP_CASES = 60          # what an investigator would work through in a shift


def money(value: float) -> str:
    """Indian numbering, because the amounts are read by people."""
    if value >= 1e7:
        return f"{value / 1e7:.2f} Cr"
    if value >= 1e5:
        return f"{value / 1e5:.2f} L"
    if value >= 1e3:
        return f"{value / 1e3:.1f} K"
    return f"{value:.0f}"


def build_queue() -> tuple[list[dict], str]:
    """Build the queue, and report which model version scored it.

    The version matters and was wrong once: the queue was scored by
    `wire_seed_xgboost_42.npy`, which came from HI-Small (v1, 55.5% recall),
    while the monitor advertised HI-Medium (v2, 69.5%). The console showed a
    headline figure directly above a queue a weaker model had produced.
    """
    scores = np.load(cfg.INTERIM / "wire_seed_xgboost_42.npy")
    version = "v1"          # HI-Small - see loaders.load_wire / cfg.AML_TRANS
    df = sort_by_time(loaders.load_wire())
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M",
                               errors="coerce")
    test_df = df.iloc[int(len(df) * 0.80):].copy()
    test_df["score"] = scores
    del df
    gc.collect()

    alerts = test_df.nlargest(int(len(test_df) * 0.01), "score")
    graph = build_temporal_graph(test_df)
    cases = build_cases(alerts, graph)
    latest = test_df["_ts"].max()

    queue = []
    for case in cases[:TOP_CASES]:
        flows = sorted(case.flows, key=lambda f: -f.hops)[:3]
        first_seen = case.transactions["_ts"].min()
        queue.append({
            "id": int(case.case_id),
            "score": round(float(case.max_score), 4),
            "value": float(case.total_value),
            "value_label": money(case.total_value),
            "accounts": len(case.accounts),
            "alerts": len(case.transactions),
            "evidence": case.evidence,
            # NaT in, NaN out - and json.dumps would happily write a bare NaN
            # literal, which is not valid JSON and renders as "NaNh".
            "age_hours": (round(age, 1)
                          if np.isfinite(age := (latest - first_seen).total_seconds() / 3600)
                          else None),
            # Ground truth is available here because this is a held-out
            # evaluation set. A live system would not have it, and the dashboard
            # shows it only so the queue's quality can be judged.
            "actually_laundering": bool(case.transactions[LABEL].sum()),
            "flows": [{
                "shape": "cycle" if f.is_cycle else "chain",
                "hops": f.hops,
                "path": f.accounts,
                "amount_in": float(f.amounts[0]),
                "amount_out": float(f.amounts[-1]),
                # out/in, not a fraction of anything. Two single-edge amounts,
                # so it exceeds 1.0 whenever the onward payment is larger than
                # what arrived - 15 of the shipped flows do, one at 20.8x. The
                # page must not label this a percentage retained.
                "out_over_in": round(float(f.retained), 3)
                if np.isfinite(f.retained) else None,
                "span_hours": round(float(f.span_hours), 1),
            } for f in flows],
            "fans": [{
                "hub": fan.hub, "direction": fan.direction,
                "counterparties": len(fan.counterparties),
                "total": float(fan.total),
                "span_hours": round(float(fan.span_hours), 1),
            } for fan in sorted(case.fans,
                                key=lambda f: -len(f.counterparties))[:2]],
        })
    return queue, version


def build_card_decisions(limit: int = 40) -> list[dict]:
    """Recent card-rail decisions - blocks and step-ups.

    The card rail produces no queue. It decides inside the authorisation window
    in about 5 ms and the transaction is allowed, challenged or declined; there
    is nothing to review afterwards in the way a wire case is reviewed.

    What an operator does want is the opposite view: **what did we decline
    today, and was it right?** A wrongly declined customer calls support, and
    this is the list that call is answered from.
    """
    scores = np.load(cfg.INTERIM / "card_scores_final.npy")
    y = np.load(cfg.INTERIM / "card_y_test.npy")
    X = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")
    amounts = X["TransactionAmt"].to_numpy(dtype="float64")

    block_t = float(np.quantile(scores, 1 - 0.005))
    alert_t = float(np.quantile(scores, 1 - 0.02))

    order = np.argsort(-scores)[:limit]
    rows = []
    for i in order:
        action = ("block" if scores[i] >= block_t
                  else "step_up" if scores[i] >= alert_t else "allow")
        rows.append({
            "id": int(i),
            "action": action,
            "score": round(float(scores[i]), 4),
            "value": float(amounts[i]),
            "value_label": money(float(amounts[i])),
            # Held-out data, so the outcome is known. A live console would show
            # this column only once a chargeback had settled.
            "actually_fraud": bool(y[i]),
        })
    return rows


def build_monitor(queue_version: str) -> dict:
    """Health of both rails, plus a drift check against a fitted baseline."""
    card_scores = np.load(cfg.INTERIM / "card_scores_final.npy")
    card_y = np.load(cfg.INTERIM / "card_y_test.npy")
    card_X = pd.read_parquet(cfg.INTERIM / "card_X_test.parquet")

    half = len(card_scores) // 2
    baseline = Baseline.build(card_scores[:half],
                              threshold=float(np.quantile(card_scores[:half], 0.98)),
                              features=card_X.iloc[:half])
    findings = check(baseline, card_scores[half:], card_X.iloc[half:])

    wire_card = json.loads((cfg.ARTIFACTS / "wire_model_card_medium.json").read_text())

    return {
        "drift": [{"metric": f.metric, "severity": f.severity,
                   "baseline": round(f.baseline, 5), "current": round(f.current, 5),
                   "detail": f.detail} for f in findings],
        "rails": {
            "card": {
                "model": "LightGBM + XGBoost, Platt-calibrated",
                "version": "v1", "alert_rate": 0.02, "block_rate": 0.005,
                "recall": 0.456, "precision": 0.784,
                "value_recall": 0.442, "latency_p99_ms": 5.53,
                "genuine_blocked": 0.00105,
                "source": "IEEE-CIS (real transactions)",
            },
            "wire": {
                "model": "XGBoost on 49 graph features",
                "version": "v2", "alert_rate": 0.01,
                "recall": wire_card["metrics"]["recall_at_operating_point"],
                "precision": wire_card["metrics"]["precision_at_operating_point"],
                "latency_p99_ms": 10.58,
                "source": "IBM AML HI-Medium (synthetic)",
                # The queue below was scored by a different version. Stated
                # rather than hidden, so the page can say so.
                "queue_version": queue_version,
                "queue_source": "IBM AML HI-Small (synthetic)",
                "queue_recall": 0.5548,
            },
        },
        "registry": json.loads((cfg.ARTIFACTS / "registry" / "registry.json").read_text())
        if (cfg.ARTIFACTS / "registry" / "registry.json").exists() else {},
    }


def main() -> None:
    print("building the queue...", flush=True)
    queue, queue_version = build_queue()
    print(f"  {len(queue)} cases (scored by wire {queue_version})", flush=True)
    if not queue:
        raise SystemExit("no cases produced - nothing to export")

    print("building the monitor...", flush=True)
    monitor = build_monitor(queue_version)

    print("building card decisions...", flush=True)
    card_decisions = build_card_decisions()

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # The queue is wire-only by construction - the card rail decides inside
        # the authorisation window and queues nothing.
        "queue": queue,
        "card_decisions": card_decisions,
        "monitor": monitor,
    }
    out = cfg.REPORTS / "dashboard_data.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwritten -> {out}  ({out.stat().st_size / 1e3:.0f} KB)")

    real = sum(c["actually_laundering"] for c in queue)
    print(f"  wire queue precision: {real}/{len(queue)} ({real / len(queue):.0%})")
    blocked = [d for d in card_decisions if d["action"] == "block"]
    if blocked:
        right = sum(d["actually_fraud"] for d in blocked)
        print(f"  card blocks shown: {len(blocked)}, {right} were fraud")
    print(f"  with traced structure: "
          f"{sum(1 for c in queue if c['flows'] or c['fans'])}/{len(queue)}")


if __name__ == "__main__":
    main()
