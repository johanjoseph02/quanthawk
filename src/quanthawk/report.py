"""Fill the tables in docs/benchmark.md from results/benchmark.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quanthawk.simulate import FRAUD_TYPES


def _ms(d: dict[str, float], pct: bool = False, digits: int = 3) -> str:
    if pct:
        return f"{100 * d['mean']:.1f} ± {100 * d['sd']:.1f}"
    return f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f}"


def main_table(res: dict[str, Any]) -> str:
    s = res["summary"]
    head = ("| Model | Average precision | ROC AUC | Recall @ 0.5% | Precision @ 0.5% | "
            "Recall @ 1% | Precision @ 1% | Episodes caught @ 1% |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for name, m in s.items():
        bold = "**" if "Quant Hawk" in name else ""
        rows.append(
            f"| {bold}{name}{bold} | {_ms(m['average_precision'])} | {_ms(m['roc_auc'])} | "
            f"{_ms(m['recall@0.5%'], True)} | {_ms(m['precision@0.5%'], True)} | "
            f"{_ms(m['recall@1.0%'], True)} | {_ms(m['precision@1.0%'], True)} | "
            f"{_ms(m['episodes_caught@1.0%'], True)} |")
    return head + "\n".join(rows) + "\n"


def by_type_table(res: dict[str, Any]) -> str:
    s = res["summary"]
    types = [t for t in FRAUD_TYPES if any(t in m["recall_by_type@1.0%"] for m in s.values())]
    head = "| Model | " + " | ".join(t.replace("_", " ") for t in types) + " |\n"
    head += "|---|" + "---:|" * len(types) + "\n"
    rows = [f"| {n} | " + " | ".join(f"{100 * m['recall_by_type@1.0%'].get(t, float('nan')):.0f}%"
                                     for t in types) + " |" for n, m in s.items()]
    return head + "\n".join(rows) + "\n"


def by_archetype_table(res: dict[str, Any]) -> str:
    s = res["summary"]
    archs = sorted(next(iter(s.values()))["false_alert_rate_by_archetype@1.0%"])
    head = "| Model | " + " | ".join(a.replace("_", " ") for a in archs) + " |\n"
    head += "|---|" + "---:|" * len(archs) + "\n"
    rows = [f"| {n} | " + " | ".join(
        f"{100 * m['false_alert_rate_by_archetype@1.0%'][a]:.2f}%" for a in archs) + " |"
        for n, m in s.items()]
    return head + "\n".join(rows) + "\n"


def segmentation_table(res: dict[str, Any]) -> str:
    head = ("| Seed | k chosen | Adjusted Rand index vs archetypes | Segments (majority archetype, "
            "share) |\n|---:|---:|---:|---|\n")
    rows = []
    for seed, r in zip(res["seeds"], res["runs"], strict=True):
        g = r["segmentation"]
        segs = "; ".join(f"{k}: {v['archetype'].replace('_', ' ')} {100 * v['share']:.0f}%"
                         for k, v in g["segment_majority_archetype"].items())
        rows.append(f"| {seed} | {g['k']} | {g['adjusted_rand_index_tx']:.2f} | {segs} |")
    return head + "\n".join(rows) + "\n"


def ablation_table(res: dict[str, Any]) -> str:
    ab = res.get("ablation_k")
    if not ab:
        return ""
    head = "| k | Average precision | Recall @ 1% | Precision @ 1% |\n|---:|---:|---:|---:|\n"
    rows = [f"| {k}{' (a single segment)' if k == '1' else ''} | {v['average_precision']:.3f} | "
            f"{100 * v['recall@1.0%']:.1f}% | {100 * v['precision@1.0%']:.1f}% |"
            for k, v in ab.items()]
    return head + "\n".join(rows) + "\n"


def homogeneous_table(res: dict[str, Any]) -> str:
    h = res.get("homogeneous")
    if not h:
        return ""
    return ("| Model | Average precision |\n|---|---:|\n"
            f"| isolation forest, global | {h['isolation forest, global']:.3f} |\n"
            f"| k-means + isolation forest (k = {h['k_chosen']}) | "
            f"{h['k-means + isolation forest']:.3f} |\n")


def data_table(res: dict[str, Any]) -> str:
    head = ("| Seed | Train transactions | Test transactions | Fraud in train | Fraud in test | "
            "Test fraud rate | Fraud episodes in test |\n|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for seed, r in zip(res["seeds"], res["runs"], strict=True):
        d = r["data"]
        ep = next(iter(r["models"].values()))["episodes"]
        rows.append(f"| {seed} | {d['train_tx']:,} | {d['test_tx']:,} | {d['train_fraud_tx']:,} | "
                    f"{d['test_fraud_tx']:,} | {100 * d['test_fraud_rate']:.3f}% | {ep} |")
    return head + "\n".join(rows) + "\n"


def write_report(results: Path, report: Path) -> None:
    res = json.loads(results.read_text())
    tables = {
        "DATA": data_table(res), "MAIN": main_table(res), "TYPES": by_type_table(res),
        "ARCHETYPES": by_archetype_table(res), "SEGMENTS": segmentation_table(res),
        "ABLATION": ablation_table(res), "HOMOGENEOUS": homogeneous_table(res),
    }
    text = report.read_text() if report.exists() else "".join(
        f"<!-- {k} -->\n<!-- /{k} -->\n" for k in tables)
    for key, table in tables.items():
        start, end = f"<!-- {key} -->", f"<!-- /{key} -->"
        if start in text and end in text:
            a, rest = text.split(start, 1)
            _, b = rest.split(end, 1)
            text = f"{a}{start}\n{table}{end}{b}"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)
