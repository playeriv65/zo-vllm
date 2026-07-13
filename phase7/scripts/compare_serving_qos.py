#!/usr/bin/env python3
"""Compare two vLLM bench serve result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    "completed",
    "failed",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "duration",
    "max_concurrent_requests",
)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compare(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        base_value = _number(base.get(metric))
        cand_value = _number(candidate.get(metric))
        if base_value is None or cand_value is None:
            out[metric] = {
                "base": base.get(metric),
                "candidate": candidate.get(metric),
                "delta": None,
                "ratio": None,
            }
            continue
        delta = cand_value - base_value
        ratio = cand_value / base_value if base_value != 0 else None
        out[metric] = {
            "base": base_value,
            "candidate": cand_value,
            "delta": delta,
            "ratio": ratio,
        }
    return out


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Serving QoS Comparison",
        "",
        f"- Base: `{summary['base_path']}`",
        f"- Candidate: `{summary['candidate_path']}`",
        "",
        "| metric | base | candidate | delta | ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, data in summary["metrics"].items():
        lines.append(
            "| {metric} | {base} | {candidate} | {delta} | {ratio} |".format(
                metric=metric,
                base=data["base"],
                candidate=data["candidate"],
                delta=data["delta"],
                ratio=data["ratio"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    base = _load(args.base)
    candidate = _load(args.candidate)
    summary = {
        "base_path": str(args.base),
        "candidate_path": str(args.candidate),
        "base_label": base.get("label"),
        "candidate_label": candidate.get("label"),
        "request_rate": {
            "base": base.get("request_rate"),
            "candidate": candidate.get("request_rate"),
        },
        "metrics": _compare(base, candidate),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, args.out_md)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
