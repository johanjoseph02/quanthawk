"""What ``qh train`` writes and ``qh serve`` loads: the model, its alert thresholds and
its drift reference, in one file.

The file is a joblib pickle. Loading a pickle runs code, so only load artefacts you
trained yourself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quanthawk import __version__
from quanthawk.drift import Reference
from quanthawk.features import FEATURES, FeatureEngine, featurize
from quanthawk.model import ModelConfig, SegmentedForest

COLUMNS = ["ts", "account_id", "channel", "amount", "merchant", "mcc", "counterparty", "city",
           "device", "account_opened"]


@dataclass
class Artifact:
    model: SegmentedForest
    thresholds: dict[str, float]  # alert budget -> score threshold on training traffic
    reference: Reference
    meta: dict[str, Any] = field(default_factory=dict)

    def threshold(self, budget: float) -> float:
        key = f"{budget:g}"
        if key in self.thresholds:
            return self.thresholds[key]
        raise KeyError(f"no threshold for budget {budget}; have {sorted(self.thresholds)}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> Artifact:
        art = joblib.load(path)
        if not isinstance(art, Artifact):
            raise TypeError(f"{path} is not a Quant Hawk artefact")
        return art


def train(history: pd.DataFrame, cfg: ModelConfig | None = None,
          budgets: tuple[float, ...] = (0.005, 0.01, 0.02)
          ) -> tuple[Artifact, FeatureEngine]:
    """Fit on ``history`` (time-ordered transactions). Returns the artefact and the
    feature engine warmed on that history, ready to score what comes next."""
    cfg = cfg or ModelConfig()
    t0 = time.time()
    X, P, H, engine = featurize(history[COLUMNS].to_dict("records"))
    profiles = np.array([p for st in engine.accounts.values() if (p := st.profile()) is not None])
    model = SegmentedForest(cfg).fit(X, P, H, profiles)
    scores, _ = model.score(X, P, H)
    thresholds = {f"{b:g}": float(np.quantile(scores, 1 - b)) for b in budgets}
    art = Artifact(model, thresholds, Reference.fit(X, FEATURES), {
        "version": __version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_tx": len(history),
        "accounts": int(history.account_id.nunique()),
        "k": model.k,
        "segment_sizes": model.segment_sizes,
        "seconds": round(time.time() - t0, 1),
    })
    return art, engine
