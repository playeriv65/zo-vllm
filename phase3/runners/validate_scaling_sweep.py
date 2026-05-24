import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.runners.collect_q1_speed_suite import (  # noqa: E402
    display_path,
    load_result,
    total_per_step,
)
from phase3.runners.collect_scaling_sweep import (  # noqa: E402
    batch_sort_key,
    discover_rows,
)


COMMON_CONFIG = {
    "steps": 1000,
    "warmup_steps": 5,
    "num_samples": 1000,
    "rank": 8,
    "lr": 3e-7,
    "eps": 1e-3,
    "step_interval": 50,
    "eval_interval": 0,
    "seed": 42,
    "zo_random_device": "cuda",
    "train_scope": "lora_only",
}

LOZO_EXPECTED_CONFIG = {
    "profile_mode": "minimal",
    "torch_compile": False,
    "torch_compile_mode": "default",
}

VLLM_EXPECTED_CONFIG = {
    "profile_mode": "detailed",
    "enforce_eager": "0",
    "batch_invariant": "0",
    "scoring_backend": "direct_worker",
    "base_eval_mode": "skip",
    "lora_residency": "gpu",
    "lora_injection": "direct",
    "lora_injection_resolved": "direct",
    "direct_lora_from_directions": "1",
    "weight_update": "direct",
    "weight_update_precision": "param",
    "qkv_weight_update": "batched",
    "sync_weight_update": "0",
    "direction_sampling": "flat",
}

VLLM_TIMING_KEYS = [
    "score_s",
    "direction_s",
    "build_lora_s",
    "lora_update_s",
    "weight_update_s",
]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_batches(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def model_batch_dir(run_dir: Path, model: str, batch: int) -> Path:
    return run_dir / f"model_{safe_model_name(model)}" / f"batch_b{batch}"


def newest_jsons(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)


def check_exactly_one(
    run_dir: Path,
    model: str,
    batch: int,
    backend: str,
    errors: list[str],
) -> tuple[Path | None, dict | None]:
    batch_dir = model_batch_dir(run_dir, model, batch)
    if backend == "lozo":
        result_dir = batch_dir / "lozo_baseline"
        pattern = "lozo_perf_minimal_*.json"
    elif backend == "vllm":
        result_dir = batch_dir / "vllm_optimized"
        pattern = "vllm_perf_detailed_*.json"
    else:
        raise ValueError(f"unknown backend: {backend}")

    paths = newest_jsons(result_dir, pattern)
    if len(paths) != 1:
        errors.append(
            f"{model} b{batch} {backend}: expected exactly one {pattern}, found {len(paths)}"
        )
        return None, None
    return paths[0], load_result(paths[0])


def check_config_values(
    label: str,
    config: dict,
    expected: dict,
    errors: list[str],
) -> None:
    for key, expected_value in expected.items():
        actual = config.get(key)
        if actual != expected_value:
            errors.append(
                f"{label}: config {key}={actual!r}, expected {expected_value!r}"
            )


def check_common_config(
    model: str,
    batch: int,
    lozo: dict,
    vllm: dict,
    errors: list[str],
) -> None:
    lozo_config = lozo.get("config", {})
    vllm_config = vllm.get("config", {})
    for key, expected in COMMON_CONFIG.items():
        lozo_value = lozo_config.get(key)
        vllm_value = vllm_config.get(key)
        if lozo_value != expected:
            errors.append(f"{model} b{batch} LOZO: {key}={lozo_value!r}, expected {expected!r}")
        if vllm_value != expected:
            errors.append(f"{model} b{batch} vLLM: {key}={vllm_value!r}, expected {expected!r}")
        if lozo_value != vllm_value:
            errors.append(
                f"{model} b{batch}: {key} mismatch, LOZO={lozo_value!r}, vLLM={vllm_value!r}"
            )
    if lozo_config.get("model_name") != model:
        errors.append(f"{model} b{batch} LOZO: model_name={lozo_config.get('model_name')!r}")
    if vllm_config.get("model_name") != model:
        errors.append(f"{model} b{batch} vLLM: model_name={vllm_config.get('model_name')!r}")
    if lozo_config.get("batch_size") != batch:
        errors.append(f"{model} b{batch} LOZO: batch_size={lozo_config.get('batch_size')!r}")
    if vllm_config.get("batch_size") != batch:
        errors.append(f"{model} b{batch} vLLM: batch_size={vllm_config.get('batch_size')!r}")


def mean_timing(timing: dict, key: str) -> float:
    value = timing.get(key)
    if not isinstance(value, dict):
        return 0.0
    return float(value.get("mean", 0.0))


def check_timing(
    model: str,
    batch: int,
    lozo: dict,
    vllm: dict,
    min_speedup: float | None,
    errors: list[str],
) -> dict:
    lozo_step = total_per_step(lozo)
    vllm_step = total_per_step(vllm)
    speedup = lozo_step / vllm_step if vllm_step > 0.0 else 0.0
    if lozo_step <= 0.0:
        errors.append(f"{model} b{batch}: non-positive LOZO total_s/step={lozo_step}")
    if vllm_step <= 0.0:
        errors.append(f"{model} b{batch}: non-positive vLLM total_s/step={vllm_step}")
    if min_speedup is not None and speedup < min_speedup:
        errors.append(
            f"{model} b{batch}: speedup {speedup:.4f} below threshold {min_speedup:.4f}"
        )

    timing = vllm.get("timing", {})
    components = {}
    for key in VLLM_TIMING_KEYS:
        components[key] = mean_timing(timing, key)
        if key != "build_lora_s" and components[key] <= 0.0:
            errors.append(f"{model} b{batch}: non-positive vLLM {key}={components[key]}")
    component_sum = sum(components.values())
    gap = vllm_step - component_sum
    if abs(gap) > 0.001:
        errors.append(
            f"{model} b{batch}: vLLM component sum gap {gap:.6f} exceeds 0.001s"
        )

    loss_change = lozo.get("loss_change")
    lozo_loss_drop = None if loss_change is None else -float(loss_change)
    return {
        "model": model,
        "batch": batch,
        "lozo_total_s_per_step": lozo_step,
        "vllm_total_s_per_step": vllm_step,
        "speedup": speedup,
        "lozo_loss_drop": lozo_loss_drop,
        "vllm_loss_drop": None,
        "vllm_component_sum_s": component_sum,
        "vllm_component_gap_s": gap,
        "vllm_score_share": components["score_s"] / vllm_step if vllm_step else 0.0,
    } | components


def validate(
    run_dir: Path,
    expected_models: list[str],
    expected_batches: list[int],
    min_speedup: float | None,
) -> dict:
    errors = []
    warnings = []
    rows = []
    artifacts = []
    if not run_dir.exists():
        raise SystemExit(f"run directory does not exist: {run_dir}")

    discovered = discover_rows(run_dir)
    discovered_pairs = sorted(
        (row["model"], row["batch"]) for row in discovered if row["lozo"] or row["vllm"]
    )
    expected_pairs = sorted((model, batch) for model in expected_models for batch in expected_batches)
    if discovered_pairs != expected_pairs:
        errors.append(
            f"model/batch set mismatch: discovered={discovered_pairs}, expected={expected_pairs}"
        )

    for model in expected_models:
        for batch in expected_batches:
            lozo_path, lozo = check_exactly_one(run_dir, model, batch, "lozo", errors)
            vllm_path, vllm = check_exactly_one(run_dir, model, batch, "vllm", errors)
            if lozo is None or vllm is None:
                continue
            lozo_config = lozo.get("config", {})
            vllm_config = vllm.get("config", {})
            if lozo_config.get("backend") != "lozo":
                errors.append(f"{model} b{batch}: LOZO backend={lozo_config.get('backend')!r}")
            if vllm_config.get("backend") != "vllm":
                errors.append(f"{model} b{batch}: vLLM backend={vllm_config.get('backend')!r}")
            check_common_config(model, batch, lozo, vllm, errors)
            check_config_values(f"{model} b{batch} LOZO", lozo_config, LOZO_EXPECTED_CONFIG, errors)
            check_config_values(f"{model} b{batch} vLLM", vllm_config, VLLM_EXPECTED_CONFIG, errors)
            if vllm_config.get("gpu_memory_utilization") == 0.5 and model != "facebook/opt-13b":
                errors.append(f"{model} b{batch}: unexpected vLLM gpu_memory_utilization=0.5")
            if vllm_config.get("gpu_memory_utilization") == 0.3 and model == "facebook/opt-13b":
                errors.append(f"{model} b{batch}: 13B vLLM unexpectedly used gpu_memory_utilization=0.3")
            if model == "facebook/opt-13b":
                warnings.append(
                    f"{model} b{batch}: vLLM uses gpu_memory_utilization="
                    f"{vllm_config.get('gpu_memory_utilization')}"
                )
            if vllm.get("loss_change") is not None:
                errors.append(f"{model} b{batch}: vLLM loss_change should be None for base_eval_mode=skip")
            rows.append(check_timing(model, batch, lozo, vllm, min_speedup, errors))
            artifacts.extend([
                display_path(lozo_path),
                display_path(vllm_path),
            ])

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "run_dir": display_path(run_dir),
        "expected_models": expected_models,
        "expected_batches": expected_batches,
        "metric": "timing.total_s / config.steps",
        "rows": sorted(rows, key=lambda row: (row["model"], batch_sort_key(Path(f"batch_b{row['batch']}")))),
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Phase 3 OPT scaling sweep.")
    parser.add_argument("run_dir")
    parser.add_argument(
        "--expected-models",
        default="facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,facebook/opt-13b",
    )
    parser.add_argument("--expected-batches", default="16,32,64,128")
    parser.add_argument("--min-speedup", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.run_dir = resolve_path(args.run_dir)
    args.expected_models = parse_models(args.expected_models)
    args.expected_batches = parse_batches(args.expected_batches)
    return args


def main() -> None:
    args = parse_args()
    report = validate(
        args.run_dir,
        args.expected_models,
        args.expected_batches,
        args.min_speedup,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text, flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
