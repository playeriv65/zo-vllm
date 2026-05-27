import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.runners.collect_q1_speed_suite import (  # noqa: E402
    display_path,
    fmt,
    load_result,
    newest_json,
    step_mean,
    total_per_step,
)


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def model_sort_key(path: Path) -> str:
    return path.name.removeprefix("model_")


def batch_sort_key(path: Path) -> int:
    try:
        return int(path.name.removeprefix("batch_b"))
    except ValueError:
        return 10**9


def load_optional(directory: Path, pattern: str) -> tuple[Path | None, dict | None]:
    path = newest_json(directory, pattern)
    if path is None:
        return None, None
    return path, load_result(path)


def loss_drop(data: dict | None) -> float | None:
    if data is None:
        return None
    value = data.get("loss_change")
    if value is None:
        return None
    return -float(value)


def gpu_util_stats(log_dir: Path, pattern: str) -> dict | None:
    paths = sorted(log_dir.glob(pattern))
    if not paths:
        return None
    path = paths[-1]
    rows_by_gpu = {}
    for row in csv.reader(path.open()):
        if len(row) < 4:
            continue
        try:
            # nvidia-smi timestamp contains a comma, so csv rows are either:
            # index, util, memory or date, time, index, util, memory.
            gpu_index = row[-3].strip()
            util = float(row[-2].strip())
            memory = float(row[-1].strip())
        except ValueError:
            continue
        rows_by_gpu.setdefault(gpu_index, []).append((util, memory))
    if not rows_by_gpu:
        return {"path": path, "samples": 0}
    gpu_index, rows = max(
        rows_by_gpu.items(),
        key=lambda item: max(memory for _, memory in item[1]) - min(memory for _, memory in item[1]),
    )
    max_memory = max(memory for _, memory in rows)
    threshold = max_memory * 0.8
    steady = [util for util, memory in rows if memory >= threshold and memory > 0]
    trimmed = steady[5:-5] if len(steady) > 10 else steady
    if not trimmed:
        trimmed = steady
    if not trimmed:
        return {"path": path, "samples": len(rows), "steady_samples": 0}
    return {
        "path": path,
        "gpu_index": gpu_index,
        "samples": len(rows),
        "steady_samples": len(steady),
        "trimmed_samples": len(trimmed),
        "mean": sum(trimmed) / len(trimmed),
        "median": statistics.median(trimmed),
        "min": min(trimmed),
        "max": max(trimmed),
    }


def row_for(model_dir: Path, batch_dir: Path) -> dict:
    lozo_path, lozo = load_optional(batch_dir / "lozo_baseline", "lozo_perf_minimal_*.json")
    vllm_path, vllm = load_optional(batch_dir / "vllm_optimized", "vllm_perf_*.json")
    lozo_step = step_mean(lozo) if lozo else None
    vllm_step = step_mean(vllm) if vllm else None
    speedup = lozo_step / vllm_step if lozo_step and vllm_step else None
    vllm_timing = vllm.get("timing", {}) if vllm else {}
    vllm_tail = vllm_timing.get("tail_100", {}) if vllm else {}
    return {
        "model": model_dir.name.removeprefix("model_").replace("__", "/"),
        "batch": batch_sort_key(batch_dir),
        "lozo_path": lozo_path,
        "vllm_path": vllm_path,
        "lozo": lozo,
        "vllm": vllm,
        "lozo_total_s_per_step": lozo_step,
        "vllm_total_s_per_step": vllm_step,
        "speedup": speedup,
        "lozo_loss_drop": loss_drop(lozo),
        "vllm_loss_drop": loss_drop(vllm),
        "vllm_score_s": vllm_tail.get("score_s", vllm_timing.get("score_s", {})).get("mean"),
        "vllm_direction_s": vllm_tail.get("direction_s", vllm_timing.get("direction_s", {})).get("mean"),
        "vllm_build_lora_s": vllm_tail.get("build_lora_s", vllm_timing.get("build_lora_s", {})).get("mean"),
        "vllm_lora_update_s": vllm_tail.get("lora_update_s", vllm_timing.get("lora_update_s", {})).get("mean"),
        "vllm_weight_update_s": vllm_tail.get("weight_update_s", vllm_timing.get("weight_update_s", {})).get("mean"),
        "vllm_weight_fold_s": vllm_tail.get("weight_fold_s", vllm_timing.get("weight_fold_s", {})).get("mean"),
        "lozo_gpu": gpu_util_stats(batch_dir / "logs", "gpu_util_lozo_*.csv"),
        "vllm_gpu": gpu_util_stats(batch_dir / "logs", "gpu_util_vllm_*.csv"),
    }


def discover_rows(run_dir: Path) -> list[dict]:
    rows = []
    for model_dir in sorted(run_dir.glob("model_*"), key=model_sort_key):
        if not model_dir.is_dir():
            continue
        for batch_dir in sorted(model_dir.glob("batch_b*"), key=batch_sort_key):
            if batch_dir.is_dir():
                rows.append(row_for(model_dir, batch_dir))
    return rows


def build_summary(run_dir: Path, rows: list[dict]) -> str:
    lines = [
        "# Phase 3 Scaling Sweep Summary",
        "",
        f"Run directory: `{display_path(run_dir)}`",
        "",
        "## Speed",
        "",
        "| model | batch | lozo_loss_drop | vllm_loss_drop | lozo_s/step | vllm_s/step | speedup | vllm_gpu_util_mean | vllm_gpu_util_median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        vllm_gpu = row["vllm_gpu"] or {}
        lines.append(
            "| {model} | {batch} | {lozo_drop} | {vllm_drop} | {lozo_step} | "
            "{vllm_step} | {speedup} | {gpu_mean} | {gpu_median} |".format(
                model=row["model"],
                batch=row["batch"],
                lozo_drop=fmt(row["lozo_loss_drop"]),
                vllm_drop=fmt(row["vllm_loss_drop"]),
                lozo_step=fmt(row["lozo_total_s_per_step"]),
                vllm_step=fmt(row["vllm_total_s_per_step"]),
                speedup=fmt(row["speedup"], 4),
                gpu_mean=fmt(vllm_gpu.get("mean")),
                gpu_median=fmt(vllm_gpu.get("median")),
            )
        )

    lines.extend([
        "",
        "## vLLM Timing",
        "",
        "| model | batch | score_s | direction_s | build_lora_s | lora_update_s | weight_update_s | weight_fold_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {model} | {batch} | {score} | {direction} | {build_lora} | "
            "{lora_update} | {weight_update} | {weight_fold} |".format(
                model=row["model"],
                batch=row["batch"],
                score=fmt(row["vllm_score_s"]),
                direction=fmt(row["vllm_direction_s"]),
                build_lora=fmt(row["vllm_build_lora_s"]),
                lora_update=fmt(row["vllm_lora_update_s"]),
                weight_update=fmt(row["vllm_weight_update_s"]),
                weight_fold=fmt(row["vllm_weight_fold_s"]),
            )
        )

    lines.extend(["", "## Artifacts", ""])
    for row in rows:
        for key in ["lozo_path", "vllm_path"]:
            if row[key] is not None:
                lines.append(f"- `{display_path(row[key])}`")
        for key in ["lozo_gpu", "vllm_gpu"]:
            gpu = row[key]
            if gpu and gpu.get("path") is not None:
                lines.append(f"- `{display_path(gpu['path'])}`")
    return "\n".join(lines) + "\n"


def write_json_summary(output: Path, rows: list[dict]) -> None:
    serializable = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key in {"lozo", "vllm"}:
                continue
            if isinstance(value, Path):
                item[key] = display_path(value)
            elif isinstance(value, dict):
                item[key] = {
                    sub_key: display_path(sub_value) if isinstance(sub_value, Path) else sub_value
                    for sub_key, sub_value in value.items()
                }
            else:
                item[key] = value
        serializable.append(item)
    output.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a Phase 3 scaling sweep.")
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    args.run_dir = resolve_path(args.run_dir)
    return args


def main() -> None:
    args = parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run directory does not exist: {args.run_dir}")
    rows = discover_rows(args.run_dir)
    if not rows:
        raise SystemExit(f"no model/batch result directories found under: {args.run_dir}")
    summary = build_summary(args.run_dir, rows)
    output = resolve_path(args.output) if args.output else args.run_dir / "summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary)
    json_output = resolve_path(args.json_output) if args.json_output else args.run_dir / "summary.json"
    write_json_summary(json_output, rows)
    print(summary)
    print(f"saved={output}", flush=True)
    print(f"json_saved={json_output}", flush=True)


if __name__ == "__main__":
    main()
