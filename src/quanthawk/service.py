"""Real-time scoring API and the analyst's alert queue.

    uv run qh serve --model models/quanthawk.joblib --warm data/history.csv.gz

Each transaction is featurised by the same ``FeatureEngine`` that built the training data,
scored by its segment's forest, and alerted if its score clears the threshold for the
configured alert budget. Alerts carry the reasons an analyst needs to triage them, and the
analyst's verdict is stored next to the alert, so precision can be measured on live
traffic.
"""

from __future__ import annotations

import hmac
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from quanthawk.artifact import COLUMNS, Artifact
from quanthawk.drift import DriftMonitor
from quanthawk.features import CHANNELS, FeatureEngine, Tx, featurize

STATIC = Path(__file__).parent / "static"

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY, tx_id TEXT, account_id INTEGER NOT NULL, ts REAL NOT NULL,
    channel TEXT NOT NULL, amount REAL NOT NULL, score REAL NOT NULL, segment INTEGER NOT NULL,
    reasons TEXT NOT NULL, created REAL NOT NULL,
    verdict TEXT, note TEXT, reviewed REAL
);
CREATE INDEX IF NOT EXISTS alerts_open ON alerts (verdict, score);
CREATE TABLE IF NOT EXISTS counters (
    segment INTEGER PRIMARY KEY, scored INTEGER NOT NULL, alerted INTEGER NOT NULL
);
"""


class TxIn(BaseModel):
    tx_id: str | None = None
    ts: float = Field(..., description="seconds since epoch")
    account_id: int
    channel: Literal["pos", "online", "atm", "transfer_out", "transfer_in"]
    amount: float = Field(..., gt=0)
    merchant: int = -1
    mcc: int = -1
    counterparty: int = -1
    city: int = -1
    device: int = 0
    account_opened: float | None = None


class Review(BaseModel):
    verdict: Literal["fraud", "legit"]
    note: str = Field("", max_length=2000)


class Scorer:
    def __init__(self, artifact: Artifact, db_path: str, budget: float = 0.01,
                 engine: FeatureEngine | None = None) -> None:
        self.art = artifact
        self.budget = budget
        self.threshold = artifact.threshold(budget)
        self.engine = engine or FeatureEngine()
        self.drift = DriftMonitor(artifact.reference)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def warm(self, history: pd.DataFrame) -> int:
        """Replay history so account baselines exist before live traffic arrives."""
        with self.lock:
            featurize(history[COLUMNS].to_dict("records"), self.engine)
        return len(history)

    def score(self, tx: TxIn) -> dict[str, Any]:
        with self.lock:
            feats, profile = self.engine.update(Tx.from_mapping(tx.model_dump()))
            has = np.array([profile is not None])
            prof = (profile if profile is not None else np.zeros(len(self.art.model.profile_names)))
            s, seg = self.art.model.score(feats[None, :], prof[None, :], has)
            score, segment = float(s[0]), int(seg[0])
            self.drift.observe(feats)
            alert = score >= self.threshold
            reasons = self.art.model.explain(feats, segment) if alert else []
            self.db.execute(
                "INSERT INTO counters (segment, scored, alerted) VALUES (?, 1, ?) "
                "ON CONFLICT(segment) DO UPDATE SET scored = scored + 1, "
                "alerted = alerted + excluded.alerted", (segment, int(alert)))
            alert_id = None
            if alert:
                cur = self.db.execute(
                    "INSERT INTO alerts (tx_id, account_id, ts, channel, amount, score, segment,"
                    " reasons, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (tx.tx_id, tx.account_id, tx.ts, tx.channel, tx.amount, score, segment,
                     json.dumps(reasons), time.time()))
                alert_id = cur.lastrowid
        return {"tx_id": tx.tx_id, "score": round(score, 4), "segment": segment,
                "threshold": round(self.threshold, 4), "alert": alert, "alert_id": alert_id,
                "reasons": reasons}

    def alerts(self, status: str, limit: int) -> list[dict[str, Any]]:
        where = {"open": "verdict IS NULL", "reviewed": "verdict IS NOT NULL", "all": "1"}[status]
        with self.lock:
            rows = self.db.execute(
                f"SELECT * FROM alerts WHERE {where} ORDER BY score DESC, id LIMIT ?",
                (limit,)).fetchall()
        return [{**dict(r), "reasons": json.loads(r["reasons"])} for r in rows]

    def review(self, alert_id: int, r: Review) -> dict[str, Any]:
        with self.lock:
            cur = self.db.execute(
                "UPDATE alerts SET verdict = ?, note = ?, reviewed = ? WHERE id = ?",
                (r.verdict, r.note, time.time(), alert_id))
            if cur.rowcount == 0:
                raise KeyError(alert_id)
            row = self.db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return {**dict(row), "reasons": json.loads(row["reasons"])}

    def stats(self) -> dict[str, Any]:
        with self.lock:
            seg = [dict(r) for r in self.db.execute("SELECT * FROM counters ORDER BY segment")]
            rev = self.db.execute(
                "SELECT COUNT(*) AS n, SUM(verdict = 'fraud') AS fraud, "
                "SUM(verdict IS NULL) AS open FROM alerts").fetchone()
        scored = sum(s["scored"] for s in seg)
        alerted = sum(s["alerted"] for s in seg)
        reviewed = (rev["n"] or 0) - (rev["open"] or 0)
        return {
            "scored": scored,
            "alerted": alerted,
            "alert_rate": alerted / scored if scored else None,
            "budget": self.budget,
            "open_alerts": rev["open"] or 0,
            "reviewed": reviewed,
            "precision_of_reviewed": (rev["fraud"] or 0) / reviewed if reviewed else None,
            "by_segment": [{**s, "alert_rate": s["alerted"] / s["scored"]} for s in seg],
            "accounts_known": len(self.engine.accounts),
            "model": self.art.meta,
        }


def create_app(scorer: Scorer, api_token: str = "") -> FastAPI:
    app = FastAPI(title="Quant Hawk", version="0.2.0")

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if not api_token:
            return
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given, api_token):
            raise HTTPException(401, "missing or wrong bearer token")

    guarded = [Depends(auth)]

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "channels": CHANNELS}

    @app.post("/v1/transactions", dependencies=guarded)
    def score(tx: TxIn) -> dict[str, Any]:
        return scorer.score(tx)

    @app.post("/v1/transactions/batch", dependencies=guarded)
    def score_batch(txs: list[TxIn]) -> list[dict[str, Any]]:
        return [scorer.score(t) for t in sorted(txs, key=lambda t: t.ts)]

    @app.get("/v1/alerts", dependencies=guarded)
    def alerts(status: Literal["open", "reviewed", "all"] = "open",
               limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[dict[str, Any]]:
        return scorer.alerts(status, limit)

    @app.post("/v1/alerts/{alert_id}/review", dependencies=guarded)
    def review(alert_id: int, body: Review) -> dict[str, Any]:
        try:
            return scorer.review(alert_id, body)
        except KeyError:
            raise HTTPException(404, f"no alert {alert_id}") from None

    @app.get("/v1/stats", dependencies=guarded)
    def stats() -> dict[str, Any]:
        return scorer.stats()

    @app.get("/v1/drift", dependencies=guarded)
    def drift() -> dict[str, Any]:
        return scorer.drift.report()

    if STATIC.is_dir():
        @app.get("/", include_in_schema=False)
        def ui() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app
