"""``qh`` command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

DAY = 86400.0


def _simulate(a: argparse.Namespace) -> int:
    from quanthawk.simulate import SimConfig, simulate

    df = simulate(SimConfig(accounts=a.accounts, days=a.days, seed=a.seed))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df):,} transactions, {int(df.is_fraud.sum())} fraudulent, written to {out}")
    return 0


def _split(path: str, train_days: int):  # type: ignore[no-untyped-def]
    import pandas as pd

    df = pd.read_csv(path)
    return df[df.ts < train_days * DAY], df[df.ts >= train_days * DAY]


def _train(a: argparse.Namespace) -> int:
    from quanthawk.artifact import train
    from quanthawk.model import ModelConfig

    history, live = _split(a.data, a.train_days)
    art, _ = train(history, ModelConfig(k=a.k, seed=a.seed))
    art.save(a.out)
    print(json.dumps(art.meta, indent=2))
    if a.split_dir:
        d = Path(a.split_dir)
        d.mkdir(parents=True, exist_ok=True)
        history.to_csv(d / "history.csv.gz", index=False)
        live.drop(columns=["is_fraud", "fraud_type", "archetype"]).to_csv(
            d / "live.csv.gz", index=False)
        live[["tx_id", "is_fraud", "fraud_type"]].to_csv(d / "live-labels.csv.gz", index=False)
        print(f"history and live traffic written to {d}")
    return 0


def _evaluate(a: argparse.Namespace) -> int:
    from quanthawk.evaluate import run

    run(tuple(a.seeds), Path(a.out), extras=not a.no_extras)
    from quanthawk.report import write_report

    write_report(Path(a.out), Path(a.report))
    return 0


def _report(a: argparse.Namespace) -> int:
    from quanthawk.report import write_report

    write_report(Path(a.results), Path(a.report))
    return 0


def _scorer(model: str, db: str, budget: float, warm: str | None):  # type: ignore[no-untyped-def]
    import pandas as pd

    from quanthawk.artifact import Artifact
    from quanthawk.service import Scorer

    art = Artifact.load(model)
    for f in art.model.forests.values():
        f.n_jobs = 1  # one row at a time: threads cost more than they save
    scorer = Scorer(art, db, budget)
    if warm:
        n = scorer.warm(pd.read_csv(warm))
        print(f"warmed {len(scorer.engine.accounts):,} accounts from {n:,} transactions")
    return scorer


def _serve(a: argparse.Namespace) -> int:
    import uvicorn

    from quanthawk.service import create_app

    scorer = _scorer(a.model, a.db, a.budget, a.warm)
    uvicorn.run(create_app(scorer, os.environ.get("QH_API_TOKEN", "")), host=a.host,
                port=a.port, log_level="warning")
    return 0


def _replay(a: argparse.Namespace) -> int:
    import httpx
    import pandas as pd

    df = pd.read_csv(a.data).sort_values("ts")
    if a.limit:
        df = df.head(a.limit)
    headers = {"Authorization": f"Bearer {a.token}"} if a.token else {}
    alerts, t0 = 0, time.time()
    with httpx.Client(base_url=a.url, headers=headers, timeout=10) as c:
        rows = df.to_dict("records")
        for i in range(0, len(rows), a.batch):
            chunk = [{k: v for k, v in r.items() if v == v} for r in rows[i:i + a.batch]]
            for r in chunk:
                r["tx_id"] = str(r.get("tx_id", ""))
            res = c.post("/v1/transactions/batch", json=chunk)
            res.raise_for_status()
            alerts += sum(x["alert"] for x in res.json())
            if a.rate:
                time.sleep(len(chunk) / a.rate)
            if (i // a.batch) % 20 == 0:
                done = i + len(chunk)
                print(f"{done:,} sent, {alerts} alerts, "
                      f"{done / max(time.time() - t0, 1e-9):,.0f} tx/s", flush=True)
    print(f"done: {len(df):,} transactions, {alerts} alerts")
    return 0


def _demo(a: argparse.Namespace) -> int:
    import tempfile

    import httpx
    import uvicorn

    from quanthawk.artifact import train
    from quanthawk.service import Scorer, create_app
    from quanthawk.simulate import SimConfig, simulate

    print("simulating a bank of 1,500 customers over 90 days...", flush=True)
    df = simulate(SimConfig(accounts=1500, days=90, seed=a.seed))
    history, live = df[df.ts < 60 * DAY], df[df.ts >= 60 * DAY]
    print(f"training on {len(history):,} transactions...", flush=True)
    art, engine = train(history)
    for f in art.model.forests.values():
        f.n_jobs = 1
    tmp = Path(tempfile.mkdtemp(prefix="quanthawk-demo-"))
    scorer = Scorer(art, str(tmp / "alerts.db"), budget=0.01, engine=engine)
    server = uvicorn.Server(uvicorn.Config(create_app(scorer), host="127.0.0.1", port=a.port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.0)
    print(f"\n  Alert queue: http://127.0.0.1:{a.port}/\n  API docs:    "
          f"http://127.0.0.1:{a.port}/docs\n\n  Streaming live traffic; Ctrl-C to stop.\n",
          flush=True)
    rows = live.drop(columns=["is_fraud", "fraud_type", "archetype"]).to_dict("records")
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{a.port}", timeout=10) as c:
            for i in range(0, len(rows), 50):
                chunk = rows[i:i + 50]
                for r in chunk:
                    r["tx_id"] = str(r["tx_id"])
                    if r.get("account_opened") != r.get("account_opened"):
                        r["account_opened"] = None
                c.post("/v1/transactions/batch", json=chunk).raise_for_status()
                time.sleep(50 / a.rate)
        print("all live traffic scored; the queue stays up. Ctrl-C to stop.")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qh", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a labelled synthetic bank to CSV")
    s.add_argument("--accounts", type=int, default=3000)
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/bank.csv.gz")
    s.set_defaults(func=_simulate)

    t = sub.add_parser("train", help="train on the first days of a transactions file")
    t.add_argument("--data", default="data/bank.csv.gz")
    t.add_argument("--train-days", type=int, default=60)
    t.add_argument("--k", type=int, default=None, help="segments; default: by silhouette")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="models/quanthawk.joblib")
    t.add_argument("--split-dir", default="data", help="also write history/live files here")
    t.set_defaults(func=_train)

    e = sub.add_parser("evaluate", help="the benchmark: simulate, train and score per seed")
    e.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    e.add_argument("--out", default="results/benchmark.json")
    e.add_argument("--report", default="docs/benchmark.md")
    e.add_argument("--no-extras", action="store_true", help="skip the k and homogeneity studies")
    e.set_defaults(func=_evaluate)

    r = sub.add_parser("report", help="rebuild docs/benchmark.md tables from results")
    r.add_argument("--results", default="results/benchmark.json")
    r.add_argument("--report", default="docs/benchmark.md")
    r.set_defaults(func=_report)

    v = sub.add_parser("serve", help="scoring API and alert queue")
    v.add_argument("--model", default="models/quanthawk.joblib")
    v.add_argument("--warm", default="data/history.csv.gz", help="history to replay first")
    v.add_argument("--db", default="alerts.db")
    v.add_argument("--budget", type=float, default=0.01)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8100)
    v.set_defaults(func=_serve)

    y = sub.add_parser("replay", help="stream a transactions file into a running server")
    y.add_argument("--data", default="data/live.csv.gz")
    y.add_argument("--url", default="http://127.0.0.1:8100")
    y.add_argument("--token", default=os.environ.get("QH_API_TOKEN", ""))
    y.add_argument("--batch", type=int, default=100)
    y.add_argument("--rate", type=float, default=0.0, help="transactions per second; 0 = max")
    y.add_argument("--limit", type=int, default=0)
    y.set_defaults(func=_replay)

    d = sub.add_parser("demo", help="simulate, train, serve and stream, in one go")
    d.add_argument("--port", type=int, default=8100)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--rate", type=float, default=400.0)
    d.set_defaults(func=_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
