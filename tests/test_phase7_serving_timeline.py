import json

import pytest

from phase7.scripts.analyze_serving_timeline import analyze


def test_analyze_timeline_detects_real_foreground_overlap(tmp_path):
    bench_path = tmp_path / "bench.json"
    zo_path = tmp_path / "zo.jsonl"
    bench_path.write_text(
        json.dumps(
            {
                "start_times": [10.0, 20.0],
                "ttfts": [1.0, 1.0],
                "itls": [[9.0], [9.0]],
                "completed": 2,
                "failed": 0,
                "request_throughput": 0.1,
                "request_goodput": 0.1,
                "p99_ttft_ms": 1000.0,
                "p99_tpot_ms": 9000.0,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "event": "train_step",
            "step": 1,
            "step_start_perf": 12.0,
            "step_end_perf": 14.0,
            "score_start_perf": 12.5,
            "score_end_perf": 13.5,
            "slot_write_start_perf": 12.1,
            "slot_write_end_perf": 12.2,
            "step_s": 2.0,
            "foreground_idle_wait_s": 0.0,
            "lora_update_s": 0.1,
            "score_s": 1.0,
        },
        {
            "event": "train_step",
            "step": 2,
            "step_start_perf": 31.0,
            "step_end_perf": 32.0,
            "score_start_perf": 31.1,
            "score_end_perf": 31.5,
            "slot_write_start_perf": 31.0,
            "slot_write_end_perf": 31.1,
        },
    ]
    zo_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    summary = analyze(bench_path, zo_path)

    assert summary["bench"]["p99_e2e_ms"] == pytest.approx(10000.0)
    assert summary["zo"]["steps_overlap_bench"] == 1
    assert summary["zo"]["steps_after_bench"] == 1
    assert summary["zo"]["steps_completed_before_bench_end"] == 1
    assert summary["zo"]["score_intervals_overlapping_requests"] == 1
    assert summary["zo"]["slot_write_intervals_overlapping_requests"] == 1
    assert summary["per_step"][0]["slot_request_overlap_count"] == 1
