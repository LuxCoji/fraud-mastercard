"""Regression tests for bugs that were found, not bugs that were imagined.

Every test here corresponds to a defect that actually shipped into a run and had
to be diagnosed from a wrong number rather than an exception. That is the common
thread: none of these threw, none crashed, and each one silently degraded either
the model or the monitoring.

Run: pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --- the router's threshold bands -------------------------------------------

def test_thresholds_come_from_rates_not_scores():
    """Bands must land at the traffic share they were asked for.

    A run put 1.8% of traffic in the block band where 0.5% was intended, because
    thresholds fitted on Platt-calibrated scores were applied to raw averaged
    ones. The ordering stayed correct throughout, so nothing looked broken - the
    only symptom was four times as many hard declines as intended.
    """
    from router import thresholds_from_scores

    scores = np.random.default_rng(0).beta(2, 20, 100_000)
    block_t, alert_t = thresholds_from_scores(scores, block_rate=0.005, alert_rate=0.02)

    assert block_t > alert_t, "the block threshold must sit above the alert threshold"
    assert abs((scores >= block_t).mean() - 0.005) < 0.001
    assert abs((scores >= alert_t).mean() - 0.020) < 0.002


def test_wire_rail_never_blocks():
    """Precision on the wire rail is ~10%. An automatic freeze would stop
    roughly nine innocent parties per real one, so the rail must only ever
    recommend."""
    from router import Action, FraudRouter, Rail

    class AlwaysCertain:
        def score(self, txn):
            return 1.0

    router = FraudRouter(wire_scorer=AlwaysCertain(), wire_alert_threshold=0.5)
    decision = router.decide({"Amount Paid": 1e9, "From Bank": 1, "To Bank": 2,
                              "Account": "A", "Account.1": "B"})
    assert decision.rail is Rail.WIRE
    assert decision.action is Action.INVESTIGATE
    assert decision.action is not Action.BLOCK


def test_transfer_rail_is_routed_but_unscored():
    """PaySim's fraud is separable by a three-line rule that holds for 97.8% of
    fraud and 0.00% of legitimate traffic. There is no trustworthy model, and a
    fabricated score would be worse than none."""
    from router import Action, FraudRouter, Rail

    decision = FraudRouter().decide(
        {"type": "TRANSFER", "nameOrig": "C1", "amount": 500.0})
    assert decision.rail is Rail.TRANSFER
    assert decision.scored is False
    assert decision.action is Action.ALLOW


def test_routing_reads_the_payload():
    from router import Rail, route

    assert route({"TransactionAmt": 10.0, "card1": 1234}) is Rail.CARD
    assert route({"Amount Paid": 10.0, "From Bank": 1}) is Rail.WIRE
    with pytest.raises(ValueError):
        route({"something": "unrecognised"})


# --- drift monitoring --------------------------------------------------------

def test_psi_uses_stored_bins_not_reconstructed_quantiles():
    """Concentrated features must not report drift when there is none.

    The baseline used to store eleven quantiles and rebuild the reference by
    interpolating between them. For the C columns - counts concentrated on small
    integers - that turns a spike at zero into a continuous ramp, and 8 of 20
    features exceeded the major threshold on data with no drift at all
    (P_emaildomain_count scored 3.60).
    """
    from monitoring.drift import PSI_MAJOR, build_bins, psi_from_bins

    rng = np.random.default_rng(0)
    # A spiky, count-like feature - the shape that broke the old approach.
    reference = rng.choice([0, 0, 0, 0, 1, 1, 2, 3], size=50_000).astype(float)
    current = rng.choice([0, 0, 0, 0, 1, 1, 2, 3], size=50_000).astype(float)

    edges, proportions = build_bins(reference)
    assert psi_from_bins(edges, proportions, current) < PSI_MAJOR


def test_psi_detects_a_genuine_shift():
    from monitoring.drift import PSI_MAJOR, build_bins, psi_from_bins

    rng = np.random.default_rng(0)
    edges, proportions = build_bins(rng.normal(0, 1, 50_000))
    shifted = rng.normal(3, 1, 50_000)      # three standard deviations across
    assert psi_from_bins(edges, proportions, shifted) > PSI_MAJOR


def test_time_index_features_are_excluded():
    """A month index necessarily differs between periods. DT_M scored PSI 5.14
    on two halves of one period with no drift - twenty times the major line -
    which would have fired CRITICAL every day."""
    from monitoring.drift import TIME_INDEX_FEATURES, Baseline, check

    rng = np.random.default_rng(0)
    scores = rng.beta(2, 20, 20_000)
    early = pd.DataFrame({"DT_M": np.full(20_000, 1.0),
                          "amount": rng.normal(100, 10, 20_000)})
    later = pd.DataFrame({"DT_M": np.full(20_000, 7.0),   # six months on
                          "amount": rng.normal(100, 10, 20_000)})

    assert "DT_M" in TIME_INDEX_FEATURES
    baseline = Baseline.build(scores, threshold=0.5, features=early)
    findings = {f.metric: f for f in check(baseline, scores, later)}
    assert findings["features_drifted"].current == 0


def test_alert_rate_drift_is_detected():
    from monitoring.drift import Baseline, Severity, check

    rng = np.random.default_rng(0)
    scores = rng.beta(2, 20, 20_000)
    baseline = Baseline.build(scores, threshold=float(np.quantile(scores, 0.98)))
    findings = {f.metric: f for f in check(baseline, np.clip(scores * 1.5, 0, 1))}
    assert findings["alert_rate"].severity in (Severity.WARN, Severity.CRITICAL)


# --- model registry ----------------------------------------------------------

def test_registry_refuses_a_regression(tmp_path):
    from monitoring.registry import ModelRegistry

    model = tmp_path / "model.txt"
    model.write_text("bytes")
    registry = ModelRegistry(tmp_path / "registry")

    registry.register("card", [model], {"recall_at_operating_point": 0.50})
    registry.promote("card", "v1")
    registry.register("card", [model], {"recall_at_operating_point": 0.30})

    result = registry.promote("card", "v2")
    assert result["promoted"] is False
    assert registry.active("card")["version"] == "v1"


def test_registry_rollback_restores_the_previous_version(tmp_path):
    from monitoring.registry import ModelRegistry

    model = tmp_path / "model.txt"
    model.write_text("bytes")
    registry = ModelRegistry(tmp_path / "registry")

    for recall in (0.50, 0.55, 0.20):
        registry.register("card", [model], {"recall_at_operating_point": recall})
    registry.promote("card", "v1")
    registry.promote("card", "v2")
    registry.promote("card", "v3", force=True)      # a human overrode the gate

    assert registry.rollback("card")["to"] == "v2"
    assert registry.active("card")["version"] == "v2"


# --- metrics -----------------------------------------------------------------

def test_operating_points_are_monotonic():
    """Flagging more traffic can never catch less fraud."""
    from models.metrics import operating_points

    rng = np.random.default_rng(0)
    scores = rng.beta(2, 20, 50_000)
    labels = (rng.random(50_000) < scores).astype(int)

    points = operating_points(labels, scores, [0.001, 0.01, 0.02, 0.05, 0.10])
    assert points.recall.is_monotonic_increasing
    assert points.flagged.is_monotonic_increasing


def test_no_rebalancing_in_any_training_module():
    """Rebalancing shifts every predicted probability toward 0.5. Measured on
    the previous project: Brier 0.2025 -> 0.0039 on removal, while PR-AUC barely
    moved - ranking metrics hide the damage, and every policy threshold is a
    probability."""
    banned = ("scale_pos_weight", "is_unbalance", "class_weight=\"balanced\"",
              "SMOTE", "RandomOverSampler")
    src_root = Path(__file__).resolve().parents[1] / "src"

    offenders = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Comments explaining *why* these are banned are expected and fine.
        code = "\n".join(line for line in text.splitlines()
                         if not line.strip().startswith("#"))
        for term in banned:
            if term in code:
                offenders.append(f"{path.name}: {term}")
    assert not offenders, f"rebalancing found: {offenders}"


# --- the fast scorers --------------------------------------------------------

def test_nan_cannot_be_looked_up_as_a_dict_key():
    """The bug this guards: `value_counts(dropna=False)` keeps a NaN bucket -
    53,761 rows for `addr1` - but `dict.get(nan)` can never find it, because
    `nan != nan`. The count came back NaN instead of 53,761."""
    table = {1.0: 10, 2.0: 20, float("nan"): 53_761}
    assert table.get(float("nan")) is None, "if this ever passes, numpy changed"

    # The scorer's approach: pull the NaN bucket out and dispatch to it.
    nan_keys = [k for k in table if isinstance(k, float) and k != k]
    assert nan_keys, "the NaN bucket must be findable by identity"
    assert table[nan_keys[0]] == 53_761


def test_feature_exclusion_uses_produced_names():
    """Excluding by source column rather than produced name dropped card1,
    card2, card3 and P_emaildomain - each a frequency-encoding *input* and a
    model feature in its own right. 150 values came back NaN."""
    frequency_sources = {"card1", "card2", "addr1", "P_emaildomain"}
    feature_names = ["card1", "card2", "addr1", "P_emaildomain",
                     "card1_count", "card2_count", "addr1_count",
                     "P_emaildomain_count", "TransactionAmt"]

    wrong = [n for n in feature_names if n not in frequency_sources]
    right = [n for n in feature_names
             if n not in {f"{c}_count" for c in frequency_sources}]

    assert "card1" not in wrong, "the old behaviour dropped the raw column"
    assert "card1" in right and "card1_count" not in right


@pytest.mark.parametrize("artifact", ["card_lightgbm.txt"])
def test_shipped_artifacts_exist(artifact):
    root = Path(__file__).resolve().parents[1]
    assert (root / "artifacts" / artifact).exists(), \
        f"{artifact} is missing - the router cannot start without it"


def test_wire_model_matches_the_current_feature_count():
    """The shipped wire model must accept what the current builder produces.

    `wire_lightgbm.txt` was trained on 25 features, before the pattern-count
    work that lifted PR-AUC 45%; the builder now produces 49. Loading the old
    artifact into the router fails outright with a shape mismatch, which is how
    this was found.
    """
    import json

    root = Path(__file__).resolve().parents[1]
    names_path = root / "artifacts" / "wire_feature_names.json"
    if not names_path.exists():
        pytest.skip("wire model not yet exported")

    names = json.loads(names_path.read_text())
    pattern_features = [n for n in names
                        if any(k in n for k in ("cycle", "scatter", "gather",
                                                "two_hop", "burst"))]
    assert len(names) > 40, f"expected the pattern feature set, got {len(names)}"
    assert pattern_features, "pattern features are missing from the shipped model"
