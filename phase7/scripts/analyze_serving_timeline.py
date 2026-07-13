#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if not _overlaps(a_start, a_end, b_start, b_end):
        return 0.0
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _request_intervals(bench: dict[str, Any]) -> list[dict[str, float | int]]:
    start_times = bench.get("start_times") or []
    ttfts = bench.get("ttfts") or []
    itls = bench.get("itls") or []
    intervals = []
    for index, start in enumerate(start_times):
        ttft = float(ttfts[index]) if index < len(ttfts) else 0.0
        per_token = itls[index] if index < len(itls) and itls[index] else []
        latency = ttft + float(sum(per_token))
        intervals.append(
            {
                "index": index,
                "start": float(start),
                "first_token": float(start) + ttft,
                "end": float(start) + latency,
                "ttft": ttft,
                "latency": latency,
            }
        )
    return intervals


def _interval_overlaps_requests(
    start: float,
    end: float,
    requests: list[dict[str, float | int]],
) -> tuple[int, float]:
    count = 0
    total = 0.0
    for request in requests:
        overlap = _overlap_s(start, end, float(request["start"]), float(request["end"]))
        if overlap > 0.0:
            count += 1
            total += overlap
    return count, total


def _mean(values: list[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not (0.0 < float(percentile) <= 100.0):
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(float(value) for value in values)
    index = max(
        0, min(len(ordered) - 1, int(len(ordered) * percentile / 100.0 + 0.999999) - 1)
    )
    return ordered[index]


def analyze(bench_path: Path, zo_path: Path) -> dict[str, Any]:
    bench = _load_json(bench_path)
    rows = _load_jsonl(zo_path)
    requests = _request_intervals(bench)
    train_rows = [row for row in rows if row.get("event") == "train_step"]
    perf_rows = [
        row for row in train_rows if "step_start_perf" in row and "step_end_perf" in row
    ]

    if not requests:
        raise ValueError("bench result does not contain detailed request intervals")
    if train_rows and not perf_rows:
        raise ValueError(
            "ZO JSONL has train rows but no perf intervals; rerun with the latest "
            "Phase7 instrumentation"
        )

    bench_start = min(float(request["start"]) for request in requests)
    bench_end = max(float(request["end"]) for request in requests)
    request_latencies_ms = [float(request["latency"]) * 1000.0 for request in requests]

    steps_before_bench = 0
    steps_after_bench = 0
    steps_overlap_bench = 0
    steps_completed_before_bench_end = 0
    step_overlap_requests = 0
    score_overlap_requests = 0
    slot_overlap_requests = 0
    slot_inside_bench_no_request_overlap = 0
    max_slot_request_overlap_s = 0.0
    max_score_request_overlap_s = 0.0
    step_s = []
    idle_wait_s = []
    lora_update_s = []
    score_s = []
    score_admission_wait_s = []
    score_admission_samples = []
    score_submit_load_samples = 0
    slot_write_load_samples = 0
    score_submit_with_foreground_load: int | None = 0
    slot_write_with_foreground_load: int | None = 0

    per_step = []
    for row in perf_rows:
        step_start = float(row["step_start_perf"])
        step_end = float(row["step_end_perf"])
        step_s.append(float(row.get("step_s", step_end - step_start)))
        idle_wait_s.append(float(row.get("foreground_idle_wait_s", 0.0)))
        lora_update_s.append(float(row.get("lora_update_s", 0.0)))
        score_s.append(float(row.get("score_s", 0.0)))
        score_admission_wait = float(row.get("score_admission_wait_s", 0.0))
        score_admission_wait_s.append(score_admission_wait)
        score_admission_samples.append(float(row.get("score_admission_samples", 0.0)))
        score_submit_load = None
        if "foreground_load_at_score_submit" in row:
            score_submit_load_samples += 1
            score_submit_load = int(row.get("foreground_load_at_score_submit") or 0)
            if score_submit_load > 0:
                assert score_submit_with_foreground_load is not None
                score_submit_with_foreground_load += 1
        slot_write_load = None
        if "foreground_load_at_slot_write" in row:
            slot_write_load_samples += 1
            slot_write_load = int(row.get("foreground_load_at_slot_write") or 0)
            if slot_write_load > 0:
                assert slot_write_with_foreground_load is not None
                slot_write_with_foreground_load += 1

        if step_end <= bench_start:
            steps_before_bench += 1
        elif step_start >= bench_end:
            steps_after_bench += 1
        else:
            steps_overlap_bench += 1
        if step_end <= bench_end:
            steps_completed_before_bench_end += 1

        step_req_count, step_req_overlap = _interval_overlaps_requests(
            step_start, step_end, requests
        )
        step_overlap_requests += int(step_req_count > 0)

        score_start = row.get("score_start_perf")
        score_end = row.get("score_end_perf")
        score_req_count = 0
        score_req_overlap = 0.0
        if score_start is not None and score_end is not None:
            score_req_count, score_req_overlap = _interval_overlaps_requests(
                float(score_start), float(score_end), requests
            )
            score_overlap_requests += int(score_req_count > 0)
            max_score_request_overlap_s = max(
                max_score_request_overlap_s, score_req_overlap
            )

        slot_start = row.get("slot_write_start_perf")
        slot_end = row.get("slot_write_end_perf")
        slot_req_count = 0
        slot_req_overlap = 0.0
        slot_inside_bench = False
        if slot_start is not None and slot_end is not None:
            slot_start_f = float(slot_start)
            slot_end_f = float(slot_end)
            slot_inside_bench = _overlaps(
                slot_start_f, slot_end_f, bench_start, bench_end
            )
            slot_req_count, slot_req_overlap = _interval_overlaps_requests(
                slot_start_f, slot_end_f, requests
            )
            slot_overlap_requests += int(slot_req_count > 0)
            max_slot_request_overlap_s = max(
                max_slot_request_overlap_s, slot_req_overlap
            )
            if slot_inside_bench and slot_req_count == 0:
                slot_inside_bench_no_request_overlap += 1

        per_step.append(
            {
                "step": int(row.get("step", -1)),
                "overlaps_bench": _overlaps(
                    step_start, step_end, bench_start, bench_end
                ),
                "step_request_overlap_count": step_req_count,
                "step_request_overlap_s": step_req_overlap,
                "score_request_overlap_count": score_req_count,
                "score_request_overlap_s": score_req_overlap,
                "slot_request_overlap_count": slot_req_count,
                "slot_request_overlap_s": slot_req_overlap,
                "slot_inside_bench": slot_inside_bench,
                "foreground_idle_wait_s": float(row.get("foreground_idle_wait_s", 0.0)),
                "score_admission_wait_s": score_admission_wait,
                "foreground_load_at_score_submit": score_submit_load,
                "foreground_load_at_slot_write": slot_write_load,
                "lora_update_s": float(row.get("lora_update_s", 0.0)),
                "score_s": float(row.get("score_s", 0.0)),
            }
        )

    return {
        "bench": {
            "path": str(bench_path),
            "num_requests": len(requests),
            "completed": bench.get("completed"),
            "failed": bench.get("failed"),
            "request_throughput": bench.get("request_throughput"),
            "request_goodput": bench.get("request_goodput"),
            "mean_ttft_ms": bench.get("mean_ttft_ms"),
            "p99_ttft_ms": bench.get("p99_ttft_ms"),
            "mean_tpot_ms": bench.get("mean_tpot_ms"),
            "p99_tpot_ms": bench.get("p99_tpot_ms"),
            "mean_e2e_ms": _mean(request_latencies_ms),
            "p99_e2e_ms": _nearest_rank_percentile(request_latencies_ms, 99.0),
            "start_perf": bench_start,
            "end_perf": bench_end,
            "duration_s": bench_end - bench_start,
        },
        "zo": {
            "path": str(zo_path),
            "train_steps": len(train_rows),
            "train_steps_with_perf": len(perf_rows),
            "steps_before_bench": steps_before_bench,
            "steps_overlap_bench": steps_overlap_bench,
            "steps_after_bench": steps_after_bench,
            "steps_completed_before_bench_end": steps_completed_before_bench_end,
            "steps_overlapping_requests": step_overlap_requests,
            "score_intervals_overlapping_requests": score_overlap_requests,
            "slot_write_intervals_overlapping_requests": slot_overlap_requests,
            "slot_writes_inside_bench_without_request_overlap": (
                slot_inside_bench_no_request_overlap
            ),
            "bench_step_density_per_s": (
                float(steps_overlap_bench / (bench_end - bench_start))
                if bench_end > bench_start
                else 0.0
            ),
            "bench_score_overlap_density_per_s": (
                float(score_overlap_requests / (bench_end - bench_start))
                if bench_end > bench_start
                else 0.0
            ),
            "slot_writes_with_foreground_load": (
                slot_write_with_foreground_load if slot_write_load_samples else None
            ),
            "slot_write_load_samples": slot_write_load_samples,
            "score_submits_with_foreground_load": (
                score_submit_with_foreground_load if score_submit_load_samples else None
            ),
            "score_submit_load_samples": score_submit_load_samples,
            "mean_step_s": _mean(step_s),
            "mean_idle_wait_s": _mean(idle_wait_s),
            "max_idle_wait_s": max(idle_wait_s) if idle_wait_s else 0.0,
            "mean_score_admission_wait_s": _mean(score_admission_wait_s),
            "max_score_admission_wait_s": (
                max(score_admission_wait_s) if score_admission_wait_s else 0.0
            ),
            "p99_score_admission_wait_s": _nearest_rank_percentile(
                score_admission_wait_s, 99.0
            ),
            "mean_score_admission_samples": _mean(score_admission_samples),
            "mean_lora_update_s": _mean(lora_update_s),
            "mean_score_s": _mean(score_s),
            "max_slot_request_overlap_s": max_slot_request_overlap_s,
            "max_score_request_overlap_s": max_score_request_overlap_s,
        },
        "per_step": per_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-json", required=True, type=Path)
    parser.add_argument("--zo-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    summary = analyze(args.bench_json, args.zo_jsonl)
    print(json.dumps({"bench": summary["bench"], "zo": summary["zo"]}, indent=2))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
