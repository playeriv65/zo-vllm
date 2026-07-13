#!/usr/bin/env python3
"""Summarize opt-in vLLM scheduler batch traces for serving-time ZO."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _kind_counts(requests: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for request in requests:
        kind = str(request.get("kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _batch_class(row: dict[str, Any]) -> str:
    counts = _kind_counts(list(row.get("requests") or []))
    has_zo = counts.get("zo", 0) > 0
    has_fg = sum(value for key, value in counts.items() if key != "zo") > 0
    if has_zo and has_fg:
        return "mixed"
    if has_zo:
        return "zo_only"
    if has_fg:
        return "foreground_only"
    return "empty"


def _compact_request(request: dict[str, Any]) -> str:
    req_id = str(request.get("req_id", ""))
    short_id = req_id
    if req_id.startswith("serving-zo-"):
        parts = req_id.split("-")
        short_id = "-".join(parts[:5])
    elif len(req_id) > 32:
        short_id = req_id[:32]
    kind = request.get("kind")
    priority = request.get("priority")
    tokens = request.get("num_scheduled_tokens")
    phase = request.get("phase")
    lora_id = request.get("lora_id")
    return (
        f"{short_id}({kind},p={priority},tok={tokens}," f"phase={phase},lora={lora_id})"
    )


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _attention_key_sum(start_pos: int, num_tokens: int) -> int:
    if num_tokens <= 0:
        return 0
    return num_tokens * (start_pos + 1) + (num_tokens * (num_tokens - 1)) // 2


def _request_token_metrics(request: dict[str, Any]) -> dict[str, int]:
    scheduled = int(request.get("num_scheduled_tokens") or 0)
    prefill = request.get("prefill_scheduled_tokens")
    decode = request.get("decode_scheduled_tokens")
    if prefill is not None and decode is not None:
        return {
            "prefill_tokens": int(prefill),
            "decode_tokens": int(decode),
            "prefill_attention_keys": int(
                request.get("prefill_attention_key_tokens") or 0
            ),
            "decode_attention_keys": int(
                request.get("decode_attention_key_tokens") or 0
            ),
        }

    num_computed = request.get("num_computed_tokens")
    num_prompt = request.get("num_prompt_tokens")
    if num_computed is None or num_prompt is None:
        inferred_prefill = scheduled if request.get("phase") == "prefill" else 0
        inferred_decode = scheduled - inferred_prefill
        return {
            "prefill_tokens": inferred_prefill,
            "decode_tokens": inferred_decode,
            "prefill_attention_keys": 0,
            "decode_attention_keys": 0,
        }

    start = int(num_computed)
    prompt_remaining = max(int(num_prompt) - start, 0)
    inferred_prefill = min(scheduled, prompt_remaining)
    inferred_decode = scheduled - inferred_prefill
    return {
        "prefill_tokens": inferred_prefill,
        "decode_tokens": inferred_decode,
        "prefill_attention_keys": _attention_key_sum(start, inferred_prefill),
        "decode_attention_keys": _attention_key_sum(
            start + inferred_prefill,
            inferred_decode,
        ),
    }


def _batch_feature_totals(row: dict[str, Any]) -> dict[str, float]:
    totals = {
        "foreground_prefill_tokens": 0.0,
        "foreground_decode_tokens": 0.0,
        "zo_prefill_tokens": 0.0,
        "zo_decode_tokens": 0.0,
        "prefill_attention_keys_k": 0.0,
        "decode_attention_keys_k": 0.0,
    }
    for request in row.get("requests") or []:
        metrics = _request_token_metrics(request)
        prefix = "zo" if request.get("kind") == "zo" else "foreground"
        totals[f"{prefix}_prefill_tokens"] += metrics["prefill_tokens"]
        totals[f"{prefix}_decode_tokens"] += metrics["decode_tokens"]
        totals["prefill_attention_keys_k"] += metrics["prefill_attention_keys"] / 1000.0
        totals["decode_attention_keys_k"] += metrics["decode_attention_keys"] / 1000.0
    return totals


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular linear system")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def _fit_linear_cost_model(
    rows: list[dict[str, Any]],
    *,
    max_target_ms: float,
) -> dict[str, Any]:
    feature_names = [
        "foreground_decode_tokens",
        "foreground_prefill_tokens",
        "zo_decode_tokens",
        "zo_prefill_tokens",
        "decode_attention_keys_k",
        "prefill_attention_keys_k",
    ]
    samples: list[dict[str, Any]] = []
    for row, next_row in zip(rows, rows[1:]):
        target_ms = (float(next_row["perf_time"]) - float(row["perf_time"])) * 1000.0
        if target_ms <= 0.0 or target_ms > max_target_ms:
            continue
        if int(row.get("total_num_scheduled_tokens") or 0) <= 0:
            continue
        samples.append(
            {
                "batch_id": row.get("batch_id"),
                "target_interval_ms": target_ms,
                "features": _batch_feature_totals(row),
            }
        )

    min_samples = len(feature_names) + 2
    if len(samples) < min_samples:
        return {
            "enabled": False,
            "reason": f"need at least {min_samples} samples, got {len(samples)}",
            "num_samples": len(samples),
            "target": "next_scheduler_batch_interval_ms",
            "max_target_ms": max_target_ms,
        }

    columns = ["intercept", *feature_names]
    xtx = [[0.0 for _ in columns] for _ in columns]
    xty = [0.0 for _ in columns]
    for sample in samples:
        x = [1.0, *[float(sample["features"][name]) for name in feature_names]]
        y = float(sample["target_interval_ms"])
        for i, xi in enumerate(x):
            xty[i] += xi * y
            for j, xj in enumerate(x):
                xtx[i][j] += xi * xj

    for i in range(1, len(columns)):
        xtx[i][i] += 1e-6

    try:
        coefficients = _solve_linear_system(xtx, xty)
    except ValueError as exc:
        return {
            "enabled": False,
            "reason": str(exc),
            "num_samples": len(samples),
            "target": "next_scheduler_batch_interval_ms",
            "max_target_ms": max_target_ms,
        }

    y_values = [float(sample["target_interval_ms"]) for sample in samples]
    y_mean = sum(y_values) / len(y_values)
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    enriched_samples = []
    residuals = []
    for sample in samples:
        x = [1.0, *[float(sample["features"][name]) for name in feature_names]]
        prediction = sum(coeff * xi for coeff, xi in zip(coefficients, x))
        target = float(sample["target_interval_ms"])
        residual = target - prediction
        residuals.append(residual)
        enriched_samples.append(
            {
                "batch_id": sample["batch_id"],
                "target_interval_ms": target,
                "predicted_interval_ms": prediction,
                "residual_ms": residual,
                "features": sample["features"],
            }
        )
    ss_res = sum(residual**2 for residual in residuals)
    rmse = math.sqrt(ss_res / len(residuals))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None

    return {
        "enabled": True,
        "target": "next_scheduler_batch_interval_ms",
        "max_target_ms": max_target_ms,
        "num_samples": len(samples),
        "feature_units": {
            "foreground_decode_tokens": "tokens",
            "foreground_prefill_tokens": "tokens",
            "zo_decode_tokens": "tokens",
            "zo_prefill_tokens": "tokens",
            "decode_attention_keys_k": "thousand key-token reads",
            "prefill_attention_keys_k": "thousand key-token reads",
        },
        "coefficients_ms": dict(zip(columns, coefficients)),
        "rmse_ms": rmse,
        "r2": r2,
        "target_p50_ms": _percentile(y_values, 50.0),
        "target_p95_ms": _percentile(y_values, 95.0),
        "target_p99_ms": _percentile(y_values, 99.0),
        "top_actual_batches": sorted(
            enriched_samples,
            key=lambda sample: sample["target_interval_ms"],
            reverse=True,
        )[:10],
        "top_predicted_batches": sorted(
            enriched_samples,
            key=lambda sample: sample["predicted_interval_ms"],
            reverse=True,
        )[:10],
    }


def analyze(trace_path: Path, *, max_target_ms: float = 1000.0) -> dict[str, Any]:
    rows = [
        row for row in _load_jsonl(trace_path) if row.get("event") == "scheduler_batch"
    ]
    class_counts = {"empty": 0, "foreground_only": 0, "zo_only": 0, "mixed": 0}
    batches_with_zo = []
    mixed_examples = []
    max_zo_per_batch = 0
    max_fg_per_batch = 0
    total_tokens = 0
    total_zo_tokens = 0
    total_fg_tokens = 0
    token_feature_totals = {
        "foreground_prefill_tokens": 0.0,
        "foreground_decode_tokens": 0.0,
        "zo_prefill_tokens": 0.0,
        "zo_decode_tokens": 0.0,
        "prefill_attention_keys_k": 0.0,
        "decode_attention_keys_k": 0.0,
    }

    for row in rows:
        batch_class = _batch_class(row)
        class_counts[batch_class] += 1
        requests = list(row.get("requests") or [])
        batch_features = _batch_feature_totals(row)
        for key, value in batch_features.items():
            token_feature_totals[key] += value
        counts = _kind_counts(requests)
        zo_count = counts.get("zo", 0)
        fg_count = sum(value for key, value in counts.items() if key != "zo")
        max_zo_per_batch = max(max_zo_per_batch, zo_count)
        max_fg_per_batch = max(max_fg_per_batch, fg_count)
        total_tokens += int(row.get("total_num_scheduled_tokens") or 0)
        for request in requests:
            tokens = int(request.get("num_scheduled_tokens") or 0)
            if request.get("kind") == "zo":
                total_zo_tokens += tokens
            else:
                total_fg_tokens += tokens
        if zo_count:
            batches_with_zo.append(int(row.get("batch_id") or -1))
        if batch_class == "mixed" and len(mixed_examples) < 20:
            mixed_examples.append(
                {
                    "batch_id": row.get("batch_id"),
                    "perf_time": row.get("perf_time"),
                    "num_requests": row.get("num_requests"),
                    "total_num_scheduled_tokens": row.get("total_num_scheduled_tokens"),
                    "token_features": batch_features,
                    "kind_counts": counts,
                    "request_order": [
                        _compact_request(request) for request in requests
                    ],
                }
            )

    return {
        "trace_path": str(trace_path),
        "total_batches": len(rows),
        "class_counts": class_counts,
        "batches_with_zo": len(batches_with_zo),
        "first_zo_batch_id": min(batches_with_zo) if batches_with_zo else None,
        "last_zo_batch_id": max(batches_with_zo) if batches_with_zo else None,
        "max_zo_requests_per_batch": max_zo_per_batch,
        "max_foreground_requests_per_batch": max_fg_per_batch,
        "total_scheduled_tokens": total_tokens,
        "total_zo_scheduled_tokens": total_zo_tokens,
        "total_foreground_scheduled_tokens": total_fg_tokens,
        "token_feature_totals": token_feature_totals,
        "cost_model": _fit_linear_cost_model(rows, max_target_ms=max_target_ms),
        "mixed_examples": mixed_examples,
    }


def write_markdown(summary: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Scheduler Trace Summary",
        "",
        f"- Trace: `{summary['trace_path']}`",
        f"- Total GPU scheduler batches: `{summary['total_batches']}`",
        f"- Batches with ZO: `{summary['batches_with_zo']}`",
        f"- First/last ZO batch: `{summary['first_zo_batch_id']}` / `{summary['last_zo_batch_id']}`",
        "",
        "## Batch Classes",
        "",
        "| class | count |",
        "|---|---:|",
    ]
    for key, value in summary["class_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Tokens",
            "",
            f"- Total scheduled tokens: `{summary['total_scheduled_tokens']}`",
            f"- Foreground scheduled tokens: `{summary['total_foreground_scheduled_tokens']}`",
            f"- ZO scheduled tokens: `{summary['total_zo_scheduled_tokens']}`",
            f"- Foreground prefill/decode tokens: "
            f"`{summary['token_feature_totals']['foreground_prefill_tokens']:.0f}` / "
            f"`{summary['token_feature_totals']['foreground_decode_tokens']:.0f}`",
            f"- ZO prefill/decode tokens: "
            f"`{summary['token_feature_totals']['zo_prefill_tokens']:.0f}` / "
            f"`{summary['token_feature_totals']['zo_decode_tokens']:.0f}`",
            "",
            "## Linear Cost Model",
            "",
        ]
    )
    model = summary["cost_model"]
    if not model.get("enabled"):
        lines.append(f"- Not fitted: `{model.get('reason')}`")
    else:
        lines.extend(
            [
                f"- Target: `{model['target']}`",
                f"- Samples: `{model['num_samples']}`",
                f"- Target p50/p95/p99: `{model['target_p50_ms']:.3f}` / "
                f"`{model['target_p95_ms']:.3f}` / "
                f"`{model['target_p99_ms']:.3f}` ms",
                f"- RMSE: `{model['rmse_ms']:.3f} ms`",
                f"- R2: `{model['r2']}`",
                "",
                "| feature | coefficient ms/unit |",
                "|---|---:|",
            ]
        )
        for name, value in model["coefficients_ms"].items():
            lines.append(f"| {name} | {value:.6f} |")
        lines.extend(["", "Top actual slow batches:", ""])
        for item in model["top_actual_batches"][:5]:
            lines.append(
                f"- batch `{item['batch_id']}`: actual "
                f"`{item['target_interval_ms']:.3f} ms`, predicted "
                f"`{item['predicted_interval_ms']:.3f} ms`, features "
                f"`{item['features']}`"
            )
    lines.extend(
        [
            "",
            "## Mixed Batch Examples",
            "",
        ]
    )
    if not summary["mixed_examples"]:
        lines.append("- No mixed foreground/ZO batches.")
    for example in summary["mixed_examples"]:
        lines.extend(
            [
                f"### Batch {example['batch_id']}",
                "",
                f"- perf_time: `{example['perf_time']}`",
                f"- num_requests: `{example['num_requests']}`",
                f"- total_num_scheduled_tokens: `{example['total_num_scheduled_tokens']}`",
                f"- token_features: `{example['token_features']}`",
                f"- kind_counts: `{example['kind_counts']}`",
                "",
                "Request order:",
                "",
            ]
        )
        for item in example["request_order"]:
            lines.append(f"- `{item}`")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument(
        "--max-target-ms",
        type=float,
        default=1000.0,
        help="Drop scheduler intervals above this value from the linear fit.",
    )
    args = parser.parse_args()

    summary = analyze(args.trace, max_target_ms=args.max_target_ms)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, args.out_md)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
