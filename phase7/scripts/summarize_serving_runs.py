#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_name(path: Path, payload: dict[str, Any]) -> str:
    bench_path = payload.get("bench", {}).get("path")
    if bench_path:
        return Path(str(bench_path)).parent.name
    return path.parent.name


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in {None, 0.0}:
        return None
    return float(value) / float(baseline)


def _pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in {None, 0.0}:
        return None
    return (float(value) / float(baseline) - 1.0) * 100.0


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(len(ordered) * float(percentile) / 100.0 + 0.999999) - 1,
        ),
    )
    return ordered[index]


def _e2e_latencies_ms(payload: dict[str, Any]) -> list[float]:
    ttfts = payload.get("ttfts") or []
    itls = payload.get("itls") or []
    latencies = []
    for index, ttft in enumerate(ttfts):
        per_token = itls[index] if index < len(itls) and itls[index] else []
        latencies.append((float(ttft) + float(sum(per_token))) * 1000.0)
    return latencies


def _bench_metric(payload: dict[str, Any], key: str) -> float | None:
    if key in payload:
        return _num(payload.get(key))
    if key == "mean_e2e_ms":
        latencies = _e2e_latencies_ms(payload)
        if latencies:
            return float(sum(latencies) / len(latencies))
    if key == "p99_e2e_ms":
        return _nearest_rank_percentile(_e2e_latencies_ms(payload), 99.0)
    return None


def _num(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def summarize(
    summary_paths: list[Path],
    *,
    baseline_path: Path | None = None,
) -> list[dict[str, Any]]:
    baseline = None if baseline_path is None else _load_json(baseline_path)
    baseline_bench = None if baseline is None else baseline.get("bench", baseline)
    rows = []
    for path in summary_paths:
        payload = _load_json(path)
        bench = payload.get("bench", {})
        zo = payload.get("zo", {})
        row = {
            "run": _run_name(path, payload),
            "summary_path": str(path),
            "completed": bench.get("completed"),
            "failed": bench.get("failed"),
            "throughput": _num(bench.get("request_throughput")),
            "goodput": _num(bench.get("request_goodput")),
            "mean_ttft_ms": _num(bench.get("mean_ttft_ms")),
            "p99_ttft_ms": _num(bench.get("p99_ttft_ms")),
            "mean_tpot_ms": _num(bench.get("mean_tpot_ms")),
            "p99_tpot_ms": _num(bench.get("p99_tpot_ms")),
            "mean_e2e_ms": _bench_metric(bench, "mean_e2e_ms"),
            "p99_e2e_ms": _bench_metric(bench, "p99_e2e_ms"),
            "zo_steps": zo.get("train_steps"),
            "zo_steps_overlap_bench": zo.get("steps_overlap_bench"),
            "zo_steps_completed_before_bench_end": zo.get(
                "steps_completed_before_bench_end"
            ),
            "zo_steps_after_bench": zo.get("steps_after_bench"),
            "zo_score_overlap": zo.get("score_intervals_overlapping_requests"),
            "zo_slot_overlap": zo.get("slot_write_intervals_overlapping_requests"),
            "zo_mean_step_s": _num(zo.get("mean_step_s")),
            "zo_mean_lora_update_s": _num(zo.get("mean_lora_update_s")),
            "zo_mean_score_s": _num(zo.get("mean_score_s")),
        }
        if baseline_bench is not None:
            row.update(
                {
                    "throughput_ratio": _ratio(
                        row["throughput"],
                        _num(baseline_bench.get("request_throughput")),
                    ),
                    "goodput_ratio": _ratio(
                        row["goodput"],
                        _num(baseline_bench.get("request_goodput")),
                    ),
                    "p99_ttft_delta_pct": _pct_delta(
                        row["p99_ttft_ms"],
                        _num(baseline_bench.get("p99_ttft_ms")),
                    ),
                    "p99_tpot_delta_pct": _pct_delta(
                        row["p99_tpot_ms"],
                        _num(baseline_bench.get("p99_tpot_ms")),
                    ),
                    "p99_e2e_delta_pct": _pct_delta(
                        row["p99_e2e_ms"],
                        _bench_metric(baseline_bench, "p99_e2e_ms"),
                    ),
                }
            )
        rows.append(row)
    return rows


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run",
        "throughput",
        "goodput",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "p99_e2e_ms",
        "zo_steps_overlap_bench",
        "zo_steps_completed_before_bench_end",
        "zo_steps_after_bench",
        "zo_mean_step_s",
        "zo_mean_lora_update_s",
        "throughput_ratio",
        "goodput_ratio",
        "p99_ttft_delta_pct",
        "p99_tpot_delta_pct",
        "p99_e2e_delta_pct",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(_format(row.get(col)) for col in columns) + " |")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    rows = summarize(args.summaries, baseline_path=args.baseline_summary)
    print_markdown(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.out_csv:
        write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()
