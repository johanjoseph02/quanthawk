"""The benchmark behind docs/benchmark.md.

    uv run qh evaluate --seeds 0 1 2

For each seed: simulate a bank (``quanthawk.simulate``), replay every transaction through
the feature engine in time order, train on the first 60 days, score the last 60.
Training data is *not* cleaned of fraud: an unsupervised model in a bank never gets that
luxury.

A fraud team works a fixed number of alerts a day, so the headline metrics are taken at
an **alert budget**: the share of transactions that can be sent to an analyst (0.5 % and
1 %). Average precision summarises the whole ranking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from quanthawk.features import DAY, FEATURES, PROFILE, featurize
from quanthawk.model import COLD, ModelConfig, SegmentedForest
from quanthawk.simulate import FRAUD_TYPES, SimConfig, simulate

BUDGETS = (0.005, 0.01)
TRAIN_DAYS = 60


# ---------------------------------------------------------------------------- data


def prepare(sim: SimConfig, train_days: int = TRAIN_DAYS) -> dict[str, Any]:
    df = simulate(sim)
    split = train_days * DAY
    train_rows = df[df.ts < split]
    test_rows = df[df.ts >= split]
    cols = ["ts", "account_id", "channel", "amount", "merchant", "mcc", "counterparty",
            "city", "device", "account_opened"]
    Xtr, Ptr, Htr, engine = featurize(train_rows[cols].to_dict("records"))
    account_profiles = np.array([p for st in engine.accounts.values()
                                 if (p := st.profile()) is not None])
    Xte, Pte, Hte, _ = featurize(test_rows[cols].to_dict("records"), engine)
    return {"df": df, "train": train_rows.reset_index(drop=True),
            "test": test_rows.reset_index(drop=True), "Xtr": Xtr, "Ptr": Ptr, "Htr": Htr,
            "Xte": Xte, "Pte": Pte, "Hte": Hte, "account_profiles": account_profiles}


# ---------------------------------------------------------------------------- metrics


def at_budget(y: NDArray[np.int64], s: NDArray[np.float64], budget: float
              ) -> tuple[NDArray[np.bool_], float, float]:
    k = max(1, round(budget * len(s)))
    idx = np.argpartition(-s, k - 1)[:k]
    alert = np.zeros(len(s), bool)
    alert[idx] = True
    tp = int((alert & (y == 1)).sum())
    return alert, tp / max(int(y.sum()), 1), tp / k


def metrics(test: pd.DataFrame, s: NDArray[np.float64]) -> dict[str, Any]:
    y = test.is_fraud.to_numpy()
    out: dict[str, Any] = {
        "average_precision": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }
    for b in BUDGETS:
        alert, recall, precision = at_budget(y, s, b)
        key = f"{b:.1%}"
        out[f"recall@{key}"] = recall
        out[f"precision@{key}"] = precision
        if b == 0.01:
            out["recall_by_type@1.0%"] = {
                t: float(alert[(test.fraud_type == t).to_numpy()].mean())
                for t in FRAUD_TYPES if (test.fraud_type == t).any()
            }
            legit = (y == 0)
            out["false_alert_rate_by_archetype@1.0%"] = {
                a: float(alert[legit & (test.archetype == a).to_numpy()].mean())
                for a in sorted(test.archetype.unique())
            }
            # episodes: a defrauded account is caught if any of its fraud is alerted
            fr = test[y == 1].assign(alert=alert[y == 1])
            ep = fr.groupby("account_id").agg(start=("ts", "min"), caught=("alert", "any"))
            first = fr[fr.alert].groupby("account_id").ts.min()
            delay = (first - ep.start.reindex(first.index)) / 60.0
            out["episodes"] = len(ep)
            out["episodes_caught@1.0%"] = float(ep.caught.mean()) if len(ep) else None
            out["median_minutes_to_first_alert@1.0%"] = (
                float(delay.median()) if len(delay) else None)
    return out


# ---------------------------------------------------------------------------- models


def rules_score(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """The kind of rule set a bank starts with, as a scored baseline."""
    f = {n: X[:, i] for i, n in enumerate(FEATURES)}
    r1 = (f["amount_z"] > 3) & ((f["new_counterparty"] > 0) | (f["new_device"] > 0))
    r2 = f["n_1h"] >= 5
    r3 = (f["new_city"] > 0) & (f["is_atm"] > 0) & (f["log_amount"] > np.log(300))
    r4 = (f["passthrough"] > 0.5) & (f["log_account_age"] < np.log1p(30))
    tie = 1 / (1 + np.exp(-f["amount_z"]))
    score: NDArray[np.float64] = r1 + r2 + r3 + r4 + 0.01 * tie
    return score


def global_forest(d: dict[str, Any], cfg: ModelConfig) -> NDArray[np.float64]:
    f = IsolationForest(n_estimators=cfg.n_estimators, max_samples=cfg.max_samples,
                        random_state=cfg.seed, n_jobs=-1).fit(d["Xtr"])
    score: NDArray[np.float64] = -f.score_samples(d["Xte"])
    return score


def segmented(d: dict[str, Any], cfg: ModelConfig) -> tuple[NDArray[np.float64], SegmentedForest]:
    m = SegmentedForest(cfg).fit(d["Xtr"], d["Ptr"], d["Htr"], d["account_profiles"])
    s, _ = m.score(d["Xte"], d["Pte"], d["Hte"])
    return s, m


def oracle_segmented(d: dict[str, Any], cfg: ModelConfig) -> NDArray[np.float64]:
    """Segments = the simulator's true archetypes. An upper bound on what better
    clustering alone could buy; not available in reality."""
    out = np.empty(len(d["Xte"]))
    tr_a = d["train"].archetype.to_numpy()
    te_a = d["test"].archetype.to_numpy()
    for a in np.unique(te_a):
        f = IsolationForest(n_estimators=cfg.n_estimators, max_samples=cfg.max_samples,
                            random_state=cfg.seed, n_jobs=-1).fit(d["Xtr"][tr_a == a])
        out[te_a == a] = -f.score_samples(d["Xte"][te_a == a])
    return out


def supervised(d: dict[str, Any], cfg: ModelConfig) -> NDArray[np.float64]:
    """Gradient boosting on the training labels: what having labels would buy."""
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         class_weight="balanced", random_state=cfg.seed)
    clf.fit(d["Xtr"], d["train"].is_fraud.to_numpy())
    p: NDArray[np.float64] = clf.predict_proba(d["Xte"])[:, 1]
    return p


def cluster_agreement(d: dict[str, Any], model: SegmentedForest) -> dict[str, Any]:
    """How the discovered segments line up with the hidden archetypes."""
    from sklearn.metrics import adjusted_rand_score

    seg = model.segment(d["Pte"], d["Hte"])
    arch = d["test"].archetype.to_numpy()
    mature = seg != COLD
    table = pd.crosstab(pd.Series(seg[mature], name="segment"),
                        pd.Series(arch[mature], name="archetype"), normalize="index")
    return {
        "k": model.k,
        "silhouette_by_k": model.silhouette,
        "adjusted_rand_index_tx": float(adjusted_rand_score(arch[mature], seg[mature])),
        "cold_share_of_test_tx": float((~mature).mean()),
        "segment_majority_archetype": {
            int(str(s)): {"archetype": str(row.idxmax()), "share": round(float(row.max()), 3)}
            for s, row in table.iterrows()
        },
    }


# ---------------------------------------------------------------------------- runs


def run_seed(seed: int, cfg: ModelConfig, sim: SimConfig | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    t0 = time.time()
    sim = replace(sim or SimConfig(), seed=seed)
    d = prepare(sim)
    test = d["test"]
    log(f"seed {seed}: {len(d['train']):,} train / {len(test):,} test transactions, "
        f"{int(test.is_fraud.sum())} fraudulent in test ({time.time() - t0:.0f}s)")
    out: dict[str, Any] = {"data": {
        "train_tx": len(d["train"]), "test_tx": len(test),
        "test_fraud_tx": int(test.is_fraud.sum()),
        "train_fraud_tx": int(d["train"].is_fraud.sum()),
        "test_fraud_rate": float(test.is_fraud.mean()),
    }, "models": {}}
    seg_s, model = segmented(d, cfg)
    scores = {
        "rules": rules_score(d["Xte"]),
        "isolation forest, global": global_forest(d, cfg),
        "k-means + isolation forest (Quant Hawk)": seg_s,
        "isolation forest per true archetype (oracle)": oracle_segmented(d, cfg),
        "gradient boosting, supervised (reference)": supervised(d, cfg),
    }
    for name, s in scores.items():
        out["models"][name] = metrics(test, s)
        log(f"  {name:48s} AP {out['models'][name]['average_precision']:.3f}  "
            f"recall@1% {out['models'][name]['recall@1.0%']:.3f}")
    out["segmentation"] = cluster_agreement(d, model)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def ablation_k(seed: int, ks: tuple[int, ...] = (1, 3, 6, 10)) -> dict[str, Any]:
    d = prepare(SimConfig(seed=seed))
    out = {}
    for k in ks:
        s, _ = segmented(d, ModelConfig(k=k, min_segment_tx=1 if k == 1 else 2000))
        m = metrics(d["test"], s)
        out[str(k)] = {key: m[key] for key in ("average_precision", "recall@1.0%",
                                               "precision@1.0%")}
    return out


def homogeneous(seed: int) -> dict[str, Any]:
    """A bank of one kind of customer: segmentation should not help here."""
    d = prepare(SimConfig(seed=seed, archetype_mix={"salaried": 1.0}))
    cfg = ModelConfig()
    seg_s, model = segmented(d, cfg)
    return {
        "k_chosen": model.k,
        "isolation forest, global": metrics(d["test"], global_forest(d, cfg))["average_precision"],
        "k-means + isolation forest": metrics(d["test"], seg_s)["average_precision"],
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    names = list(runs[0]["models"])
    keys = [k for k, v in runs[0]["models"][names[0]].items() if isinstance(v, int | float)]
    out: dict[str, Any] = {}
    for n in names:
        out[n] = {}
        for k in keys:
            vals = [r["models"][n][k] for r in runs if r["models"][n][k] is not None]
            out[n][k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
        for k in ("recall_by_type@1.0%", "false_alert_rate_by_archetype@1.0%"):
            groups = runs[0]["models"][n][k]
            out[n][k] = {g: float(np.mean([r["models"][n][k].get(g, np.nan) for r in runs]))
                         for g in groups}
    return out


def run(seeds: tuple[int, ...], out_path: Path, cfg: ModelConfig | None = None,
        sim: SimConfig | None = None, extras: bool = True) -> dict[str, Any]:
    cfg = cfg or ModelConfig()
    runs = [run_seed(s, cfg, sim) for s in seeds]
    res: dict[str, Any] = {
        "config": {"model": asdict(cfg), "sim": asdict(sim or SimConfig()),
                   "train_days": TRAIN_DAYS, "budgets": BUDGETS,
                   "features": FEATURES, "profile": PROFILE},
        "seeds": list(seeds), "runs": runs, "summary": summarise(runs),
    }
    if extras:
        print("ablation over k (seed 0)")
        res["ablation_k"] = ablation_k(seeds[0])
        print("homogeneous population (seed 0)")
        res["homogeneous"] = homogeneous(seeds[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    return res
