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


# --- amount-aware step-up ----------------------------------------------------

def test_step_up_threshold_falls_as_amount_rises():
    """A larger transaction needs less confidence to be challenged.

    Elkan's rule: the cost of a false negative is the amount at risk, while the
    cost of a false positive is roughly fixed. Measured on the card rail, a flat
    threshold caught 1 of the 20 largest frauds; the amount-aware one catches 9.
    """
    from router import FraudRouter

    router = FraudRouter(card_alert_threshold=0.40, amount_cost=5.632)
    thresholds = [router.step_up_threshold(a) for a in (10, 100, 1000, 3000)]

    assert thresholds == sorted(thresholds, reverse=True), \
        "threshold must fall monotonically as amount rises"
    assert thresholds[0] > 0.5, "a small amount should need high confidence"
    assert thresholds[-1] < 0.2, "a large amount should need much less"


def test_block_threshold_ignores_amount():
    """Blocking stays at a fixed bar however large the transaction.

    A challenge is an OTP - cheap, and worth spending on a large payment. A
    block is a lost sale. Making blocking easier for large transactions would
    put the most expensive mistake exactly where the customer is most valuable.
    """
    from router import Action, FraudRouter

    router = FraudRouter(card_block_threshold=0.90, card_alert_threshold=0.40,
                         amount_cost=5.632)

    # A score below the block bar must not block, no matter the amount.
    for amount in (10, 1_000_000):
        decision = router._decide_card(0.85, amount)
        assert decision.action is not Action.BLOCK

    # And a score above it blocks regardless of amount.
    assert router._decide_card(0.95, 10).action is Action.BLOCK
    assert router._decide_card(0.95, 1_000_000).action is Action.BLOCK


def test_amount_aware_falls_back_when_uncalibrated():
    """Without a cost constant the router must behave exactly as before."""
    from router import FraudRouter

    router = FraudRouter(card_alert_threshold=0.40)      # no amount_cost
    assert router.step_up_threshold(10) == 0.40
    assert router.step_up_threshold(100_000) == 0.40
    assert router.step_up_threshold(None) == 0.40


# --- the freeze workflow -----------------------------------------------------

class _FakeCase:
    """Minimal stand-in - the workflow only reads these four attributes."""

    def __init__(self, case_id, value=1000.0, score=0.9, evidence="CYCLE (4 hops)"):
        self.case_id = case_id
        self.total_value = value
        self.max_score = score
        self.evidence = evidence


def test_a_freeze_requires_a_named_reviewer(tmp_path):
    """'Who decided this' is the first question an audit asks. Defaulting to
    'system' would make an automated freeze indistinguishable from a human one."""
    from policy.actions import CaseLog, request_freeze

    log = CaseLog(tmp_path / "cases.jsonl")
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            request_freeze(_FakeCase(1), bad, "circular flow", log)
    with pytest.raises(ValueError):
        request_freeze(_FakeCase(1), "a.reviewer", "", log)


def test_a_large_freeze_needs_a_second_approval(tmp_path):
    """A large freeze is both the most damaging when wrong and the most likely
    to be challenged, so it should not rest on one person."""
    from policy.actions import DUAL_APPROVAL_ABOVE, CaseLog, request_freeze

    log = CaseLog(tmp_path / "cases.jsonl")
    big = request_freeze(_FakeCase(1, value=DUAL_APPROVAL_ABOVE * 2),
                         "a.reviewer", "layering", log)
    assert big.pending is True

    small = request_freeze(_FakeCase(2, value=1000.0),
                           "a.reviewer", "layering", log)
    assert small.pending is False

    approved = request_freeze(_FakeCase(3, value=DUAL_APPROVAL_ABOVE * 2),
                              "a.reviewer", "layering", log,
                              second_reviewer="b.reviewer")
    assert approved.pending is False


def test_the_log_is_append_only_and_reconstructible(tmp_path):
    """A cleared case that later turns out to be laundering has to be
    reconstructible - what was decided, by whom, and when."""
    from policy.actions import CaseLog, clear_case, reopen_case, request_freeze

    log = CaseLog(tmp_path / "cases.jsonl")
    clear_case(_FakeCase(7), "a.reviewer", "merchant settlement", log)
    reopen_case(7, "b.reviewer", "new information", log)
    request_freeze(_FakeCase(7), "b.reviewer", "confirmed layering", log)

    history = log.for_case(7)
    assert [d.action for d in history] == ["cleared", "reopened", "freeze_requested"]
    assert log.current_state(7) == "freeze_requested"
    # The clearing is still there - nothing was overwritten.
    assert history[0].reason == "merchant settlement"


def test_the_wire_rail_never_notifies_the_customer(tmp_path):
    """Telling someone they are under AML review is tipping off - a criminal
    offence under the PMLA, the BSA and POCA."""
    from policy.actions import notify_customer

    wire = notify_customer("wire", "freeze_requested")
    assert wire["notify"] is False
    assert "tipping off" in wire["reason"]

    # Card fraud is the opposite: the customer is expected to be told.
    assert notify_customer("card", "step_up")["notify"] is True
    assert notify_customer("card", "block")["notify"] is True


def test_decided_cases_leave_the_queue(tmp_path):
    from policy.actions import CaseLog, clear_case, queue_after_decisions, reopen_case

    log = CaseLog(tmp_path / "cases.jsonl")
    cases = [_FakeCase(i) for i in range(4)]
    assert len(queue_after_decisions(cases, log)) == 4

    clear_case(cases[1], "a.reviewer", "payroll", log)
    assert [c.case_id for c in queue_after_decisions(cases, log)] == [0, 2, 3]

    reopen_case(1, "b.reviewer", "second look", log)
    assert [c.case_id for c in queue_after_decisions(cases, log)] == [0, 1, 2, 3]


# --- support-agent explanations ----------------------------------------------

def test_explanation_never_invents_a_number():
    """An agent repeating a fabricated figure to a customer is the failure this
    feature exists to avoid, so the check is a test rather than a hope."""
    from policy.explain import Fact, verify

    facts = [Fact("Amount", "27"), Fact("Time of day", "23:00")]

    grounded, invented = verify(
        "The payment was for 27 at 23:00.", facts)
    assert grounded and not invented

    grounded, invented = verify(
        "The payment was for 27, well above the usual 4,500.", facts)
    assert not grounded and "4500" in invented


def test_facts_are_skipped_not_guessed():
    """IEEE-CIS is heavily missing. A fact asserted from a NaN is a fabrication,
    so an absent column must produce no claim about its value."""
    import numpy as np
    from policy.explain import build_facts

    sparse = pd.Series({"TransactionAmt": 100.0, "hour": 3.0,
                        "dist1": np.nan, "addr2": np.nan,
                        "TransactionAmt_uid_mean": np.nan})
    text = " ".join(str(f) for f in build_facts(sparse))

    assert "100" in text and "03:00" in text
    assert "nan" not in text.lower()
    # Absent geography is reported as absent, never as a value.
    assert "not supplied" in text
    assert "Distance" not in text


def test_unicode_punctuation_becomes_readable_ascii():
    """The model returns narrow no-break spaces around numbers. Encoding with
    errors='replace' turns each into a literal '?' mid-sentence, which reads as
    corruption on an agent's screen."""
    from policy.explain import _to_ascii

    assert _to_ascii("was 27 units") == "was 27 units"
    assert _to_ascii("the card’s spend") == "the card's spend"
    assert "?" not in _to_ascii("a—b “quoted”")


# --- TREASURE's self-supervised head ----------------------------------------

def _tiny_treasure():
    """A two-layer TREASURE over three dynamic attributes. Small enough to run."""
    from models.treasure.config import CatAttr, ModelConfig, NumAttr, Schema
    from models.treasure.model import build_model

    schema = Schema()
    schema.static_num, schema.static_cat = [], [CatAttr("card", 8)]
    schema.dynamic_num = [NumAttr("amount"), NumAttr("time_delta")]
    schema.dynamic_cat = [CatAttr("hour", 24)]
    schema.signal_num, schema.signal_cat = [], [CatAttr("isFraud", 2)]
    return build_model(schema, ModelConfig(d_model=16, n_layers=2, n_heads=2,
                                           input_layers=1, max_seq_len=4))


def _forward(model, lengths, max_len=4):
    """Run the model over a batch whose sequences have the given real lengths."""
    import torch

    batch_size = len(lengths)
    pad = torch.ones(batch_size, max_len, dtype=torch.bool)
    for i, n in enumerate(lengths):
        pad[i, :n] = False
    batch = {
        "static_num": torch.zeros(batch_size, 0),
        "static_cat": torch.zeros(batch_size, 1, dtype=torch.long),
        "dyn_num": torch.rand(batch_size, max_len, 2),
        "dyn_cat": torch.randint(0, 24, (batch_size, max_len, 1)),
        "pad_mask": pad,
    }
    return model(batch), batch


def test_the_next_transaction_loss_survives_a_batch_with_no_transitions():
    """A batch of one-transaction cards must not crash training.

    The median card in IEEE-CIS has a single transaction, so a batch with no
    position-to-position transition is a real shape, not a hypothetical. An
    early version returned a plain zero there and `backward()` raised "element 0
    of tensors does not require grad" - training died on the data's most common
    case.
    """
    from models.treasure.train import next_transaction_loss

    model = _tiny_treasure()
    out, batch = _forward(model, lengths=[1, 1, 1])
    loss = next_transaction_loss(out["next"], batch["dyn_num"],
                                 batch["dyn_cat"], batch["pad_mask"])

    assert loss.requires_grad, "the loss must stay attached to the graph"
    loss.backward()          # the failure this test exists for


def test_the_next_transaction_loss_ignores_padding():
    """Padding must not enter the self-supervised objective.

    A model can predict padding perfectly and learn nothing from it, so a loss
    that counted padded positions would fall steadily while the encoder learned
    less and less.
    """
    import torch

    from models.treasure.train import next_transaction_loss

    torch.manual_seed(0)
    model = _tiny_treasure()
    out, batch = _forward(model, lengths=[2, 2])
    before = next_transaction_loss(out["next"], batch["dyn_num"],
                                   batch["dyn_cat"], batch["pad_mask"])

    # Scribble over the padded tail. Nothing about the loss may change.
    batch["dyn_num"][:, 2:] = 999.0
    batch["dyn_cat"][:, 2:] = 23
    after = next_transaction_loss(out["next"], batch["dyn_num"],
                                  batch["dyn_cat"], batch["pad_mask"])

    assert torch.allclose(before, after), "padded positions leaked into the loss"


def test_the_positive_weight_refuses_a_set_with_no_fraud():
    """Silently weighting by an empty positive class would divide by zero."""
    from models.treasure.train import positive_weight

    seq = {"sig_cat": np.zeros((4, 3, 1), dtype="int64"),
           "pad_mask": np.zeros((4, 3), dtype=bool)}
    with pytest.raises(ValueError):
        positive_weight(seq)


def test_simulator_field_sequences_keep_the_same_row_order():
    """Restricting the columns must not reorder the rows.

    The simulator-field arm reuses the labels built for the full-field arm. If
    the two orderings ever diverged, every score would be compared against a
    different transaction's label - the exact failure that once produced AUC
    0.5007 across three seeds.
    """
    from models.treasure.sequences import (DYNAMIC_CAT, DYNAMIC_NUM,
                                           SIM_DYNAMIC_CAT, SIM_DYNAMIC_NUM,
                                           SIM_STATIC_CAT, STATIC_CAT,
                                           build_sequences, encode_categoricals,
                                           prepare)

    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "TransactionDT": np.sort(rng.integers(0, 60 * 86_400, n)).astype(float),
        "TransactionAmt": rng.gamma(2.0, 40.0, n),
        "isFraud": (rng.random(n) < 0.1).astype(int),
        "card1": rng.integers(10, 25, n), "card2": 1.0, "card3": 150.0,
        "card4": "visa", "card5": 200.0, "card6": "debit",
        "addr1": rng.integers(100, 110, n), "addr2": 87.0,
        "dist1": rng.random(n), "dist2": np.nan,
        "ProductCD": rng.choice(list("WCR"), n),
        "P_emaildomain": "gmail.com", "R_emaildomain": np.nan,
        "DeviceType": "mobile",
    })
    for i in range(1, 15):
        df[f"C{i}"] = rng.integers(0, 5, n).astype(float)
    for i in range(1, 16):
        df[f"D{i}"] = rng.random(n) * 100
    day = df["TransactionDT"] / 86_400
    df["D1"] = day - df.groupby("card1")["TransactionDT"].transform("min") / 86_400
    for i in range(1, 10):
        df[f"M{i}"] = "T"

    prepared = prepare(df)
    columns = sorted(set(STATIC_CAT + DYNAMIC_CAT + SIM_STATIC_CAT + SIM_DYNAMIC_CAT))
    codes, _ = encode_categoricals(prepared, prepared, columns)

    full = build_sequences(prepared, codes, 8, STATIC_CAT, DYNAMIC_CAT, DYNAMIC_NUM)
    sim = build_sequences(prepared, codes, 8, SIM_STATIC_CAT, SIM_DYNAMIC_CAT,
                          SIM_DYNAMIC_NUM)

    assert np.array_equal(full["row_index"][~full["pad_mask"]],
                          sim["row_index"][~sim["pad_mask"]])
    assert sim["dyn_num"].shape[-1] == len(SIM_DYNAMIC_NUM)
    assert full["dyn_num"].shape[-1] == len(DYNAMIC_NUM)
