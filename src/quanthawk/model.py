"""Segment customers with K-means, then learn "normal" separately for each segment.

An Isolation Forest scores a transaction by how few random splits it takes to isolate
it. Trained on everyone at once, it learns that *large* is unusual, so it fills the alert
queue with small businesses paying suppliers and wealthy customers buying furniture,
while a student's account being emptied in small transfers looks ordinary. Clustering
accounts by behaviour first and training one forest per cluster lets each forest learn
what is unusual *for that kind of customer*.

Accounts too new to have a behavioural profile (``features.MIN_PROFILE_TX``) are scored
by a forest trained on everyone, and move to their segment once they have history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from quanthawk.features import FEATURES, PROFILE

COLD = -1


@dataclass(frozen=True)
class ModelConfig:
    k: int | None = None  # None: choose by silhouette over k_range
    k_range: tuple[int, int] = (3, 10)
    n_estimators: int = 200
    max_samples: int = 256
    min_segment_tx: int = 2000  # smaller segments fall back to the global forest
    seed: int = 0


@dataclass
class SegmentedForest:
    config: ModelConfig = field(default_factory=ModelConfig)
    scaler: StandardScaler | None = None
    kmeans: KMeans | None = None
    forests: dict[int, IsolationForest] = field(default_factory=dict)
    reference: dict[int, dict[str, NDArray[np.float64]]] = field(default_factory=dict)
    silhouette: dict[int, float] = field(default_factory=dict)
    segment_sizes: dict[int, int] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURES
    profile_names: tuple[str, ...] = PROFILE

    # ------------------------------------------------------------------ fit

    def fit(self, X: NDArray[np.float64], profiles: NDArray[np.float64],
            has_profile: NDArray[np.bool_], account_profiles: NDArray[np.float64]
            ) -> SegmentedForest:
        """``X``, ``profiles``, ``has_profile``: one row per training transaction, the
        profile being the account's as it stood at that moment. ``account_profiles``: one
        row per mature account at the end of the training window, used to fit K-means."""
        cfg = self.config
        self.scaler = StandardScaler().fit(account_profiles)
        Z = self.scaler.transform(account_profiles)
        rng = np.random.default_rng(cfg.seed)
        sample = Z if len(Z) <= 3000 else Z[rng.choice(len(Z), 3000, replace=False)]
        if cfg.k is None:
            best_k, best = cfg.k_range[0], -1.0
            for k in range(cfg.k_range[0], cfg.k_range[1] + 1):
                km = KMeans(k, n_init=10, random_state=cfg.seed).fit(Z)
                s = float(silhouette_score(sample, km.predict(sample)))
                self.silhouette[k] = round(s, 4)
                if s > best:
                    best_k, best = k, s
            k = best_k
        else:
            k = cfg.k
        self.kmeans = KMeans(k, n_init=10, random_state=cfg.seed).fit(Z)

        seg = self.segment(profiles, has_profile)
        self.forests[COLD] = self._forest(X)
        self.reference[COLD] = _reference(X)
        self.segment_sizes[COLD] = int((seg == COLD).sum())
        for s in range(k):
            mask = seg == s
            self.segment_sizes[s] = int(mask.sum())
            if mask.sum() >= cfg.min_segment_tx:
                self.forests[s] = self._forest(X[mask])
                self.reference[s] = _reference(X[mask])
        return self

    def _forest(self, X: NDArray[np.float64]) -> IsolationForest:
        cfg = self.config
        return IsolationForest(n_estimators=cfg.n_estimators, max_samples=cfg.max_samples,
                               random_state=cfg.seed, n_jobs=-1).fit(X)

    # ------------------------------------------------------------------ use

    @property
    def k(self) -> int:
        assert self.kmeans is not None
        return int(self.kmeans.n_clusters)

    def segment(self, profiles: NDArray[np.float64], has_profile: NDArray[np.bool_]
                ) -> NDArray[np.int64]:
        assert self.scaler is not None and self.kmeans is not None
        out = np.full(len(profiles), COLD, np.int64)
        if has_profile.any():
            out[has_profile] = self.kmeans.predict(self.scaler.transform(profiles[has_profile]))
        return out

    def score(self, X: NDArray[np.float64], profiles: NDArray[np.float64],
              has_profile: NDArray[np.bool_]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Anomaly score (higher is more anomalous, roughly 0.3 to 0.8) and segment."""
        seg = self.segment(profiles, has_profile)
        out = np.empty(len(X))
        for s in np.unique(seg):
            mask = seg == s
            forest = self.forests.get(int(s), self.forests[COLD])
            out[mask] = -forest.score_samples(X[mask])
        return out, seg

    def explain(self, x: NDArray[np.float64], segment: int, top: int = 3) -> list[dict[str, Any]]:
        """The features on which this transaction sits furthest outside its segment."""
        ref = self.reference.get(segment, self.reference[COLD])
        z = (x - ref["median"]) / ref["spread"]
        order = np.argsort(-np.abs(z))
        names = self.feature_names
        new_in = float(x[names.index("new_in_24h_vs_max")]) if "new_in_24h_vs_max" in names else 0.0
        out = []
        for i in order:
            name = names[i]
            if name.startswith("is_") or abs(z[i]) < 2:
                continue
            # Passing on most of a few euros from a new payer is not a mule pattern.
            if name == "passthrough" and new_in < -1.0:
                continue
            typical = float(ref["median"][i])
            out.append({"feature": name, "value": float(x[i]), "typical": typical,
                        "p99": float(ref["p99"][i]), "z": round(float(z[i]), 1),
                        "text": describe(name, float(x[i]), float(ref["p99"][i]), typical)})
            if len(out) == top:
                break
        return out


def _reference(X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    q = np.percentile(X, [1, 25, 50, 75, 99], axis=0)
    spread = np.maximum(q[3] - q[1], (q[4] - q[0]) / 6, )
    return {"median": q[2], "spread": np.maximum(spread, 0.05), "p99": q[4]}


def describe(name: str, value: float, p99: float, typical: float | None = None) -> str:
    """One line an analyst can read without knowing the feature names."""
    usual = "" if typical is None else f" (typical for this segment: {typical:.0f})"
    if name == "amount_z":
        return f"amount far above this customer's usual for this channel ({value:+.1f} sd)"
    if name == "amount_vs_max":
        return f"{np.exp(value):.1f}x the largest amount this customer has used here"
    if name == "log_amount":
        return f"large amount ({np.exp(value):,.0f})"
    if name == "n_1h":
        return f"{value:.0f} transactions in the last hour{usual}"
    if name == "n_24h":
        return f"{value:.0f} transactions in the last 24 hours{usual}"
    if name == "out_24h_vs_max":
        return (f"last 24 hours' payments out {np.exp(value):.1f}x the largest this "
                "customer had made")
    if name == "new_in_24h_vs_max":
        return (f"money from new payers in the last 24 hours {np.exp(value):.1f}x the largest "
                "payment this customer had received")
    if name == "passthrough":
        return f"{value:.0%} of the last day's money from new payers already sent on"
    if name == "log_gap":
        return f"{np.expm1(value) / 60:,.0f} minutes since the previous transaction"
    if name == "new_device":
        return "device never used on this account"
    if name == "new_counterparty":
        return "payee or merchant never used on this account"
    if name == "new_city":
        return "city never seen on this account"
    if name == "new_mcc":
        return "merchant category new to this customer"
    if name == "new_cp_24h":
        return f"{value:.0f} new payees or merchants in the last 24 hours{usual}"
    if name == "hour_rarity":
        return "unusual time of day for this customer"
    if name == "log_account_age":
        return f"account {np.expm1(value):.0f} days old"
    return name
