#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(
        0, min(len(ordered) - 1, int(len(ordered) * percentile / 100.0 + 0.999999) - 1)
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


def _bench_metrics(payload: dict[str, Any]) -> dict[str, float | int | None]:
    bench = payload.get("bench", payload)
    e2e_latencies = _e2e_latencies_ms(bench)
    mean_e2e = None
    if e2e_latencies:
        mean_e2e = float(sum(e2e_latencies) / len(e2e_latencies))
    return {
        "completed": bench.get("completed"),
        "failed": bench.get("failed"),
        "request_throughput": _num(bench.get("request_throughput")),
        "request_goodput": _num(bench.get("request_goodput")),
        "p99_ttft_ms": _num(bench.get("p99_ttft_ms")),
        "p99_tpot_ms": _num(bench.get("p99_tpot_ms")),
        "mean_e2e_ms": _num(bench.get("mean_e2e_ms", mean_e2e)),
        "p99_e2e_ms": _num(
            bench.get("p99_e2e_ms", _nearest_rank_percentile(e2e_latencies, 99.0))
        ),
    }


def _num(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in {None, 0.0}:
        return None
    return float(value) / float(baseline)


def _delta_pct(value: float | None, baseline: float | None) -> float | None:
    ratio = _ratio(value, baseline)
    if ratio is None:
        return None
    return (ratio - 1.0) * 100.0


def _check_min(name: str, value: float | None, threshold: float) -> dict[str, Any]:
    passed = value is not None and value >= threshold
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "op": ">=",
        "passed": bool(passed),
    }


def _check_max(name: str, value: float | None, threshold: float) -> dict[str, Any]:
    passed = value is not None and value <= threshold
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "op": "<=",
        "passed": bool(passed),
    }


def check_acceptance(
    *,
    baseline: dict[str, Any],
    zo_summary: dict[str, Any],
    min_throughput_ratio: float,
    min_goodput_ratio: float,
    max_p99_latency_delta_pct: float,
    min_overlap_steps: int,
) -> dict[str, Any]:
    baseline_bench = _bench_metrics(baseline)
    zo_bench = _bench_metrics(zo_summary)
    zo = zo_summary.get("zo", {})

    throughput_ratio = _ratio(
        zo_bench["request_throughput"],
        baseline_bench["request_throughput"],
    )
    goodput_ratio = _ratio(
        zo_bench["request_goodput"],
        baseline_bench["request_goodput"],
    )
    p99_ttft_delta_pct = _delta_pct(
        zo_bench["p99_ttft_ms"],
        baseline_bench["p99_ttft_ms"],
    )
    p99_tpot_delta_pct = _delta_pct(
        zo_bench["p99_tpot_ms"],
        baseline_bench["p99_tpot_ms"],
    )
    p99_e2e_delta_pct = _delta_pct(
        zo_bench["p99_e2e_ms"],
        baseline_bench["p99_e2e_ms"],
    )
    overlap_steps = int(zo.get("steps_overlap_bench") or 0)
    completed_before_bench_end = int(zo.get("steps_completed_before_bench_end") or 0)
    score_overlap = int(zo.get("score_intervals_overlapping_requests") or 0)
    slot_overlap = int(zo.get("slot_write_intervals_overlapping_requests") or 0)
    train_steps = int(zo.get("train_steps") or 0)

    checks = [
        _check_max("baseline_failed_requests", baseline_bench["failed"], 0.0),
        _check_max("zo_failed_requests", zo_bench["failed"], 0.0),
        _check_min("zo_train_steps", train_steps, 1),
        _check_min("throughput_ratio", throughput_ratio, min_throughput_ratio),
        _check_min("goodput_ratio", goodput_ratio, min_goodput_ratio),
        _check_max(
            "p99_ttft_delta_pct",
            p99_ttft_delta_pct,
            max_p99_latency_delta_pct,
        ),
        _check_max(
            "p99_tpot_delta_pct",
            p99_tpot_delta_pct,
            max_p99_latency_delta_pct,
        ),
        _check_max("p99_e2e_delta_pct", p99_e2e_delta_pct, max_p99_latency_delta_pct),
        _check_min("zo_steps_overlap_bench", overlap_steps, min_overlap_steps),
        _check_min(
            "zo_steps_completed_before_bench_end",
            completed_before_bench_end,
            min_overlap_steps,
        ),
        _check_min("zo_score_overlap", score_overlap, min_overlap_steps),
        _check_min("zo_slot_overlap", slot_overlap, min_overlap_steps),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "baseline": baseline_bench,
        "zo": {
            **zo_bench,
            "train_steps": train_steps,
            "steps_overlap_bench": overlap_steps,
            "steps_completed_before_bench_end": completed_before_bench_end,
            "score_intervals_overlapping_requests": score_overlap,
            "slot_write_intervals_overlapping_requests": slot_overlap,
        },
    }


def print_report(result: dict[str, Any]) -> None:
    print(f"accepted={str(result['passed']).lower()}")
    for check in result["checks"]:
        value = check["value"]
        value_s = "" if value is None else f"{float(value):.6g}"
        print(
            f"{check['name']}: {value_s} {check['op']} "
            f"{check['threshold']} -> {check['passed']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", required=True, type=Path)
    parser.add_argument("--zo-summary-json", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.98)
    parser.add_argument("--min-goodput-ratio", type=float, default=0.98)
    parser.add_argument("--max-p99-latency-delta-pct", type=float, default=5.0)
    parser.add_argument("--min-overlap-steps", type=int, default=1)
    args = parser.parse_args()

    result = check_acceptance(
        baseline=_load_json(args.baseline_json),
        zo_summary=_load_json(args.zo_summary_json),
        min_throughput_ratio=args.min_throughput_ratio,
        min_goodput_ratio=args.min_goodput_ratio,
        max_p99_latency_delta_pct=args.max_p99_latency_delta_pct,
        min_overlap_steps=args.min_overlap_steps,
    )
    print_report(result)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
