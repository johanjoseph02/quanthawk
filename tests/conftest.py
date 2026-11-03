from __future__ import annotations

import pandas as pd
import pytest

from quanthawk.simulate import SimConfig, simulate


@pytest.fixture(scope="session")
def small_bank() -> pd.DataFrame:
    return simulate(SimConfig(accounts=300, days=60, seed=11, fraud_share=0.1))
