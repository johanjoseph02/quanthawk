from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quanthawk.artifact import Artifact, train
from quanthawk.cli import build_parser
from quanthawk.drift import Reference
from quanthawk.evaluate import at_budget, metrics, rules_score
from quanthawk.features import DAY, FEATURES, featurize
from quanthawk.model import COLD, ModelConfig, SegmentedForest, describe
from quanthawk.service import Scorer, create_app

FAST = ModelConfig(k=3, n_estimators=50, min_segment_tx=200)


@pytest.fixture(scope="module")
def trained(small_bank: pd.DataFrame) -> tuple[Artifact, pd.DataFrame, pd.DataFrame]:
    history = small_bank[small_bank.ts < 40 * DAY]
    live = small_bank[small_bank.ts >= 40 * DAY]
    art, _ = train(history, FAST)
    return art, history, live


def test_segmented_forest_scores_and_segments(trained) -> None:  # type: ignore[no-untyped-def]
    art, _, _ = trained
    assert art.model.k == 3
    assert set(art.thresholds) == {"0.005", "0.01", "0.02"}
    assert art.threshold(0.005) >= art.threshold(0.01) >= art.threshold(0.02)
    X = np.random.default_rng(0).normal(size=(5, len(FEATURES)))
    s, seg = art.model.score(X, np.zeros((5, len(art.model.profile_names))), np.zeros(5, bool))
    assert s.shape == (5,) and (seg == COLD).all()


def test_k_is_chosen_by_silhouette(small_bank: pd.DataFrame) -> None:
    X, P, H, engine = featurize(small_bank.head(20000).to_dict("records"))
    prof = np.array([p for st in engine.accounts.values() if (p := st.profile()) is not None])
    m = SegmentedForest(ModelConfig(k=None, k_range=(2, 4), n_estimators=20)).fit(X, P, H, prof)
    assert set(m.silhouette) == {2, 3, 4}
    assert m.k == max(m.silhouette, key=lambda k: m.silhouette[k])


def test_explanations_are_readable(trained) -> None:  # type: ignore[no-untyped-def]
    art, _, _ = trained
    x = np.array(art.model.reference[COLD]["median"], copy=True)
    x[FEATURES.index("n_1h")] = 12
    x[FEATURES.index("new_device")] = 1
    reasons = art.model.explain(x, COLD)
    texts = " / ".join(r["text"] for r in reasons)
    assert "12 transactions in the last hour" in texts
    text = describe("passthrough", 0.9, 0.1)
    assert text == "90% of the last day's money from new payers already sent on"


def test_artifact_round_trip(tmp_path: Path, trained) -> None:  # type: ignore[no-untyped-def]
    art, _, _ = trained
    art.save(tmp_path / "m.joblib")
    again = Artifact.load(tmp_path / "m.joblib")
    assert again.thresholds == art.thresholds and again.model.k == art.model.k


def test_drift_reference() -> None:
    rng = np.random.default_rng(0)
    ref = Reference.fit(rng.normal(size=(5000, 2)), ("a", "b"))
    same = ref.psi(rng.normal(size=(5000, 2)))
    shifted = ref.psi(rng.normal(loc=[0, 1.5], size=(5000, 2)))
    assert same["a"] < 0.02 and same["b"] < 0.02
    assert shifted["b"] > 0.25 and shifted["a"] < 0.02


def test_budget_metrics() -> None:
    y = np.array([0] * 98 + [1, 1])
    s = np.arange(100, dtype=float)
    alert, recall, precision = at_budget(y, s, 0.02)
    assert alert.sum() == 2 and recall == 1.0 and precision == 1.0
    _, recall, _ = at_budget(y, -s, 0.02)
    assert recall == 0.0


def test_metrics_and_rules_on_real_features(small_bank: pd.DataFrame) -> None:
    X, _, _, _ = featurize(small_bank.to_dict("records"))
    s = rules_score(X)
    m = metrics(small_bank, s)
    assert 0 <= m["average_precision"] <= 1
    assert set(m["recall_by_type@1.0%"]) <= {"account_takeover", "card_testing", "atm_cashout",
                                              "mule", "low_and_slow"}
    assert m["episodes"] > 0


# ---------------------------------------------------------------------------- service


@pytest.fixture
def client(tmp_path: Path, trained) -> tuple[TestClient, pd.DataFrame]:  # type: ignore[no-untyped-def]
    art, history, live = trained
    scorer = Scorer(art, str(tmp_path / "a.db"), budget=0.02)
    scorer.warm(history)
    return TestClient(create_app(scorer, api_token="s3cret")), live


def _body(row: dict) -> dict:
    keep = ("ts", "account_id", "channel", "amount", "merchant", "mcc", "counterparty", "city",
            "device")
    return {"tx_id": str(row["tx_id"]), **{k: row[k] for k in keep}}


def test_service_scores_alerts_and_takes_reviews(client) -> None:  # type: ignore[no-untyped-def]
    c, live = client
    h = {"Authorization": "Bearer s3cret"}
    assert c.post("/v1/transactions", json=_body(live.iloc[0].to_dict())).status_code == 401
    batch = [_body(r) for r in live.head(1500).to_dict("records")]
    res = c.post("/v1/transactions/batch", json=batch, headers=h).json()
    assert len(res) == 1500
    flagged = [r for r in res if r["alert"]]
    assert flagged and all(r["reasons"] is not None for r in flagged)
    alerts = c.get("/v1/alerts", headers=h).json()
    assert len(alerts) == len(flagged)
    assert alerts == sorted(alerts, key=lambda a: -a["score"])
    first = alerts[0]["id"]
    rv = c.post(f"/v1/alerts/{first}/review", json={"verdict": "fraud", "note": "called"},
                headers=h).json()
    assert rv["verdict"] == "fraud"
    missing = c.post("/v1/alerts/999999/review", json={"verdict": "legit"}, headers=h)
    assert missing.status_code == 404
    stats = c.get("/v1/stats", headers=h).json()
    assert stats["scored"] == 1500 and stats["reviewed"] == 1
    assert stats["precision_of_reviewed"] == 1.0
    assert c.get("/v1/drift", headers=h).json()["rows"] == 1500
    assert "Quant Hawk" in c.get("/").text or "QUANT HAWK" in c.get("/").text


def test_service_validates_input(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    h = {"Authorization": "Bearer s3cret"}
    bad = {"ts": 1.0, "account_id": 1, "channel": "cheque", "amount": 5}
    assert c.post("/v1/transactions", json=bad, headers=h).status_code == 422
    neg = {"ts": 1.0, "account_id": 1, "channel": "pos", "amount": -5}
    assert c.post("/v1/transactions", json=neg, headers=h).status_code == 422


def test_cli_parses() -> None:
    p = build_parser()
    for argv in (["simulate"], ["train"], ["evaluate", "--seeds", "0"], ["report"], ["serve"],
                 ["replay"], ["demo"]):
        assert p.parse_args(argv).func
