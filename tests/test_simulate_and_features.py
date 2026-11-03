from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quanthawk.artifact import COLUMNS
from quanthawk.features import (
    DAY,
    FEATURES,
    HOUR,
    MIN_PROFILE_TX,
    PROFILE,
    FeatureEngine,
    Tx,
    featurize,
)
from quanthawk.simulate import FRAUD_TYPES, SimConfig, simulate


def test_simulation_is_deterministic_and_ordered(small_bank: pd.DataFrame) -> None:
    again = simulate(SimConfig(accounts=300, days=60, seed=11, fraud_share=0.1))
    pd.testing.assert_frame_equal(small_bank, again)
    assert small_bank.ts.is_monotonic_increasing
    assert (small_bank.tx_id == range(len(small_bank))).all()
    assert small_bank.ts.max() < 60 * DAY


def test_simulation_has_every_fraud_type_and_late_accounts(small_bank: pd.DataFrame) -> None:
    assert set(small_bank.fraud_type.unique()) - {""} == set(FRAUD_TYPES)
    assert (small_bank.is_fraud == (small_bank.fraud_type != "")).all()
    late = small_bank[small_bank.account_opened > 0]
    assert len(late) and (late.ts >= late.account_opened).all()


def test_archetype_mix_override() -> None:
    df = simulate(SimConfig(accounts=50, days=10, seed=1, archetype_mix={"retiree": 1.0}))
    assert set(df.archetype) == {"retiree"}


def _rows(df: pd.DataFrame) -> list[dict]:
    return df[COLUMNS].to_dict("records")


def test_features_are_point_in_time(small_bank: pd.DataFrame) -> None:
    """A transaction's features must not depend on anything that happens after it."""
    cut = small_bank.ts.quantile(0.5)
    early = small_bank[small_bank.ts < cut]
    X_early, P_early, H_early, _ = featurize(_rows(early))
    X_all, P_all, H_all, _ = featurize(_rows(small_bank))
    n = len(early)
    np.testing.assert_allclose(X_all[:n], X_early)
    np.testing.assert_allclose(P_all[:n], P_early)
    assert (H_all[:n] == H_early).all()


def test_online_and_batch_paths_agree(small_bank: pd.DataFrame) -> None:
    rows = _rows(small_bank.head(3000))
    X, _, _, _ = featurize(rows)
    engine = FeatureEngine()
    online = np.vstack([engine.update(Tx.from_mapping(r))[0] for r in rows])
    np.testing.assert_allclose(online, X)
    assert X.shape[1] == len(FEATURES)


def f(vec: np.ndarray, name: str) -> float:
    return float(vec[FEATURES.index(name)])


def test_novelty_velocity_and_passthrough() -> None:
    e = FeatureEngine()
    base = {"account_id": 1, "city": 3, "device": 7}
    for i in range(10):
        e.update(Tx(ts=i * DAY, channel="online", amount=20, merchant=5, mcc=2, **base))
    x, _ = e.update(Tx(ts=10 * DAY, channel="online", amount=20, merchant=5, mcc=2, **base))
    assert f(x, "new_device") == 0 and f(x, "new_counterparty") == 0 and abs(f(x, "amount_z")) < 1
    x, _ = e.update(Tx(ts=10 * DAY + 60, channel="online", amount=900, merchant=99, mcc=9,
                       account_id=1, city=8, device=55))
    assert f(x, "new_device") == 1 and f(x, "new_counterparty") == 1
    assert f(x, "new_city") == 1 and f(x, "new_mcc") == 1
    assert f(x, "amount_z") > 3 and f(x, "amount_vs_max") > 3
    assert f(x, "n_1h") == 2
    # money in from a stranger, most of it straight back out
    e.update(Tx(ts=11 * DAY, channel="transfer_in", amount=1000, counterparty=500, **base))
    x, _ = e.update(Tx(ts=11 * DAY + HOUR, channel="transfer_out", amount=900,
                       counterparty=501, **base))
    assert f(x, "passthrough") == pytest.approx(0.9)
    assert f(x, "new_in_24h_vs_max") == pytest.approx(math.log1p(1000))  # no earlier inflow


def test_payday_is_ordinary_for_a_salaried_customer() -> None:
    e = FeatureEngine()
    for month in range(3):
        e.update(Tx(ts=month * 30 * DAY, account_id=9, channel="transfer_in", amount=3000,
                    counterparty=1))
        for d in range(1, 29):
            e.update(Tx(ts=(month * 30 + d) * DAY, account_id=9, channel="pos", amount=25,
                        merchant=d % 5))
    e.update(Tx(ts=90 * DAY, account_id=9, channel="transfer_in", amount=3050, counterparty=1))
    x, _ = e.update(Tx(ts=90 * DAY + HOUR, account_id=9, channel="pos", amount=25, merchant=1))
    # this month's salary from the usual payer looks like any day without new money
    assert f(x, "new_in_24h_vs_max") == pytest.approx(-math.log1p(3000))
    # the same inflow on an account that has only ever received small amounts
    for d in range(20):
        e.update(Tx(ts=d * DAY, account_id=10, channel="transfer_in", amount=40, counterparty=2))
    e.update(Tx(ts=21 * DAY, account_id=10, channel="transfer_in", amount=3050, counterparty=3))
    x, _ = e.update(Tx(ts=21 * DAY + HOUR, account_id=10, channel="pos", amount=25, merchant=1))
    assert f(x, "new_in_24h_vs_max") > 4


def test_profile_needs_history() -> None:
    e = FeatureEngine()
    for i in range(MIN_PROFILE_TX + 1):
        _, p = e.update(Tx(ts=i * DAY / 2, account_id=2, channel="pos", amount=10, merchant=i))
    assert p is not None and p.shape == (len(PROFILE),)
    assert e.accounts[2].profile() is not None
    _, p_new = e.update(Tx(ts=0, account_id=3, channel="pos", amount=10))
    assert p_new is None


def test_unknown_channel_is_rejected() -> None:
    with pytest.raises(ValueError, match="channel"):
        FeatureEngine().update(Tx(ts=0, account_id=1, channel="cheque", amount=1))
