import pytest

from phase7.scripts.check_serving_acceptance import check_acceptance


def _baseline():
    return {
        "request_throughput": 10.0,
        "request_goodput": 8.0,
        "p99_ttft_ms": 100.0,
        "p99_tpot_ms": 20.0,
        "p99_e2e_ms": 1000.0,
        "completed": 100,
        "failed": 0,
    }


def _zo_summary(*, throughput: float = 9.9, overlap: int = 3, failed: int = 0):
    return {
        "bench": {
            "request_throughput": throughput,
            "request_goodput": 7.9,
            "p99_ttft_ms": 104.0,
            "p99_tpot_ms": 20.5,
            "p99_e2e_ms": 1040.0,
            "completed": 100,
            "failed": failed,
        },
        "zo": {
            "train_steps": overlap,
            "steps_overlap_bench": overlap,
            "steps_completed_before_bench_end": overlap,
            "score_intervals_overlapping_requests": overlap,
            "slot_write_intervals_overlapping_requests": overlap,
        },
    }


def test_serving_acceptance_passes_when_ratios_and_overlap_match():
    result = check_acceptance(
        baseline=_baseline(),
        zo_summary=_zo_summary(),
        min_throughput_ratio=0.98,
        min_goodput_ratio=0.98,
        max_p99_latency_delta_pct=5.0,
        min_overlap_steps=1,
    )

    assert result["passed"] is True
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["throughput_ratio"]["value"] == pytest.approx(0.99)


def test_serving_acceptance_fails_when_background_did_not_overlap():
    result = check_acceptance(
        baseline=_baseline(),
        zo_summary=_zo_summary(overlap=0),
        min_throughput_ratio=0.98,
        min_goodput_ratio=0.98,
        max_p99_latency_delta_pct=5.0,
        min_overlap_steps=1,
    )

    assert result["passed"] is False
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    assert "zo_steps_overlap_bench" in failed


def test_serving_acceptance_fails_when_no_background_steps_ran():
    summary = _zo_summary(overlap=3)
    summary["zo"]["train_steps"] = 0
    result = check_acceptance(
        baseline=_baseline(),
        zo_summary=summary,
        min_throughput_ratio=0.98,
        min_goodput_ratio=0.98,
        max_p99_latency_delta_pct=5.0,
        min_overlap_steps=1,
    )

    assert result["passed"] is False
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    assert "zo_train_steps" in failed


def test_serving_acceptance_fails_when_background_completes_after_benchmark():
    summary = _zo_summary(overlap=3)
    summary["zo"]["steps_completed_before_bench_end"] = 0
    result = check_acceptance(
        baseline=_baseline(),
        zo_summary=summary,
        min_throughput_ratio=0.98,
        min_goodput_ratio=0.98,
        max_p99_latency_delta_pct=5.0,
        min_overlap_steps=1,
    )

    assert result["passed"] is False
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    assert "zo_steps_completed_before_bench_end" in failed


def test_serving_acceptance_fails_when_throughput_regresses():
    result = check_acceptance(
        baseline=_baseline(),
        zo_summary=_zo_summary(throughput=9.0),
        min_throughput_ratio=0.98,
        min_goodput_ratio=0.98,
        max_p99_latency_delta_pct=5.0,
        min_overlap_steps=1,
    )

    assert result["passed"] is False
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    assert "throughput_ratio" in failed


def test_serving_acceptance_fails_when_zo_requests_failed():
    result = check_acceptance(
        baseline=_baseline(),
        zo_summary=_zo_summary(failed=1),
        min_throughput_ratio=0.98,
        min_goodput_ratio=0.98,
        max_p99_latency_delta_pct=5.0,
        min_overlap_steps=1,
    )

    assert result["passed"] is False
    failed = [check["name"] for check in result["checks"] if not check["passed"]]
    assert "zo_failed_requests" in failed
