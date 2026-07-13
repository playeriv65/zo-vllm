import json

import pytest

from phase7.scripts.summarize_serving_runs import summarize


def _summary_payload(*, throughput: float, goodput: float):
    return {
        "bench": {
            "path": "phase7/logs/serving/run_a/baseline_rps2.json",
            "completed": 10,
            "failed": 0,
            "request_throughput": throughput,
            "request_goodput": goodput,
            "mean_ttft_ms": 10.0,
            "p99_ttft_ms": 100.0,
            "mean_tpot_ms": 5.0,
            "p99_tpot_ms": 20.0,
            "mean_e2e_ms": 300.0,
            "p99_e2e_ms": 400.0,
        },
        "zo": {
            "train_steps": 4,
            "steps_overlap_bench": 3,
            "steps_completed_before_bench_end": 2,
            "steps_after_bench": 1,
            "score_intervals_overlapping_requests": 3,
            "slot_write_intervals_overlapping_requests": 3,
            "mean_step_s": 0.1,
            "mean_lora_update_s": 0.02,
            "mean_score_s": 0.03,
        },
    }


def test_summarize_serving_runs_with_baseline_ratios(tmp_path):
    baseline_path = tmp_path / "baseline_summary.json"
    summary_path = tmp_path / "timeline_summary.json"
    baseline = _summary_payload(throughput=2.0, goodput=1.5)["bench"]
    baseline.pop("mean_e2e_ms")
    baseline.pop("p99_e2e_ms")
    baseline["ttfts"] = [0.1, 0.2]
    baseline["itls"] = [[0.1], [0.2]]
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(throughput=1.8, goodput=1.2)),
        encoding="utf-8",
    )

    rows = summarize([summary_path], baseline_path=baseline_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["run"] == "run_a"
    assert row["throughput_ratio"] == pytest.approx(0.9)
    assert row["goodput_ratio"] == pytest.approx(0.8)
    assert row["p99_e2e_delta_pct"] == pytest.approx(0.0)
    assert row["zo_steps_overlap_bench"] == 3
    assert row["zo_steps_completed_before_bench_end"] == 2
    assert row["zo_steps_after_bench"] == 1
