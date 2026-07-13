import json
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = PROJECT_ROOT / "phase7" / "scripts" / "analyze_scheduler_trace.py"
spec = importlib.util.spec_from_file_location("phase7_analyze_scheduler_trace", ANALYZER_PATH)
assert spec is not None and spec.loader is not None
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
analyze = analyzer.analyze


def _request(
    *,
    req_id: str,
    kind: str,
    tokens: int,
    computed: int,
    prompt: int,
) -> dict:
    return {
        "req_id": req_id,
        "kind": kind,
        "priority": 1000 if kind == "zo" else 0,
        "lora_id": 9001 if kind == "zo" else None,
        "num_scheduled_tokens": tokens,
        "num_computed_tokens": computed,
        "num_prompt_tokens": prompt,
        "num_tokens": max(prompt, computed + tokens),
        "phase": "prefill" if computed < prompt else "decode",
    }


def _row(batch_id: int, perf_time: float, requests: list[dict]) -> dict:
    return {
        "event": "scheduler_batch",
        "batch_id": batch_id,
        "perf_time": perf_time,
        "num_requests": len(requests),
        "total_num_scheduled_tokens": sum(
            int(request["num_scheduled_tokens"]) for request in requests
        ),
        "token_budget_remaining": 8192,
        "max_num_scheduled_tokens": 8192,
        "requests": requests,
    }


def test_scheduler_trace_summary_fits_simple_cost_model(tmp_path):
    perf_time = 100.0
    rows = []
    request_sets = [
        [
            _request(
                req_id="bench-a", kind="foreground", tokens=1, computed=64, prompt=32
            )
        ],
        [
            _request(
                req_id="bench-b", kind="foreground", tokens=32, computed=0, prompt=32
            )
        ],
        [
            _request(
                req_id="serving-zo-step-1-plus",
                kind="zo",
                tokens=48,
                computed=0,
                prompt=48,
            )
        ],
        [
            _request(
                req_id="bench-c", kind="foreground", tokens=1, computed=96, prompt=32
            ),
            _request(
                req_id="serving-zo-step-2-minus",
                kind="zo",
                tokens=64,
                computed=0,
                prompt=64,
            ),
        ],
        [
            _request(
                req_id="bench-d", kind="foreground", tokens=2, computed=128, prompt=32
            )
        ],
        [
            _request(
                req_id="serving-zo-step-3-plus",
                kind="zo",
                tokens=80,
                computed=0,
                prompt=80,
            )
        ],
        [
            _request(
                req_id="bench-e", kind="foreground", tokens=40, computed=0, prompt=40
            )
        ],
        [
            _request(
                req_id="bench-f", kind="foreground", tokens=1, computed=160, prompt=32
            ),
            _request(
                req_id="serving-zo-step-4-minus",
                kind="zo",
                tokens=96,
                computed=0,
                prompt=96,
            ),
        ],
        [
            _request(
                req_id="bench-g", kind="foreground", tokens=1, computed=192, prompt=32
            )
        ],
    ]
    for batch_id, requests in enumerate(request_sets, start=1):
        rows.append(_row(batch_id, perf_time, requests))
        total_tokens = rows[-1]["total_num_scheduled_tokens"]
        perf_time += (4.0 + 0.05 * total_tokens) / 1000.0
    rows.append(_row(10, perf_time, request_sets[0]))

    trace_path = tmp_path / "scheduler_trace.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = analyze(trace_path)

    assert summary["class_counts"]["mixed"] == 2
    assert summary["token_feature_totals"]["zo_prefill_tokens"] == 288
    assert summary["token_feature_totals"]["foreground_decode_tokens"] == 7
    assert summary["cost_model"]["enabled"] is True
    assert "zo_prefill_tokens" in summary["cost_model"]["coefficients_ms"]
    assert summary["mixed_examples"][0]["token_features"]["zo_prefill_tokens"] == 64
