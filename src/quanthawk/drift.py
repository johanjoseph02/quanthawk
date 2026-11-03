"""Feature drift: is live traffic still shaped like the data the model learned from?

Population stability index (PSI) per feature, between the training distribution and
a window of recently scored transactions. Common rule of thumb: below 0.1 stable, 0.1 to
0.25 worth a look, above 0.25 the model is scoring a population it has not seen.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

EPS = 1e-4


@dataclass
class Reference:
    names: tuple[str, ...]
    edges: list[NDArray[np.float64]]
    expected: list[NDArray[np.float64]]

    @classmethod
    def fit(cls, X: NDArray[np.float64], names: tuple[str, ...], bins: int = 10) -> Reference:
        edges, expected = [], []
        for j in range(X.shape[1]):
            e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
            edges.append(e)
            expected.append(_shares(X[:, j], e))
        return cls(names, edges, expected)

    def psi(self, X: NDArray[np.float64]) -> dict[str, float]:
        out = {}
        for j, name in enumerate(self.names):
            a = _shares(X[:, j], self.edges[j])
            e = self.expected[j]
            out[name] = round(float(np.sum((a - e) * np.log((a + EPS) / (e + EPS)))), 4)
        return out


def _shares(x: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.float64]:
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return counts / max(len(x), 1)


@dataclass
class DriftMonitor:
    reference: Reference
    window: int = 20000
    recent: deque[NDArray[np.float64]] = field(default_factory=deque)

    def observe(self, x: NDArray[np.float64]) -> None:
        self.recent.append(x)
        if len(self.recent) > self.window:
            self.recent.popleft()

    def report(self, min_rows: int = 500) -> dict[str, object]:
        n = len(self.recent)
        if n < min_rows:
            return {"rows": n, "status": f"waiting for {min_rows} transactions", "psi": {}}
        psi = self.reference.psi(np.vstack(self.recent))
        worst = max(psi.values()) if psi else 0.0
        status = "stable" if worst < 0.1 else ("watch" if worst < 0.25 else "drifted")
        return {"rows": n, "status": status, "worst": worst,
                "psi": dict(sorted(psi.items(), key=lambda kv: -kv[1]))}
