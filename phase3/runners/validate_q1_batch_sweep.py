import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.runners.collect_q1_batch_sweep import (  # noqa: E402
    discover_batch_dirs,
    row_for_batch,
)
from phase3.runners.collect_q1_speed_suite import (  # noqa: E402
    display_path,
    step_mean,
    total_per_step,
)


COMMON_CONFIG_KEYS = [
    "steps",
    "batch_size",
    "num_samples",
    "rank",
    "lr",
    "eps",
    "step_interval",
    "eval_interval",
    "seed",
    "zo_random_device",
    "train_scope",
]

VLLM_EXPECTED_CONFIG = {
    "batch_invariant": "0",
    "enforce_eager": "0",
    "lora_residency": "gpu",
    "lora_injection_resolved": "direct",
    "weight_update": "direct",
    "weight_update_precision": "param",
}

DETAILED_TIMING_KEYS = [
    "score_s",
    "score_generate_s",
    "score_request_build_s",
    "score_postprocess_s",
    "lora_update_s",
    "weight_update_s",
    "build_lora_s",
    "direction_s",
]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_int_list(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    items = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        items.append(int(item))
    return items


def loss_drop(data: dict) -> float:
    return -float(data.get("loss_change", 0.0))


def check_required(row: dict, errors: list[str]) -> bool:
    batch = row["batch"]
    ok = True
    for key in ["lozo", "vllm"]:
        if row[key] is None:
            errors.append(f"batch {batch}: missing required {key} result")
            ok = False
    return ok


def check_configs(row: dict, errors: list[str], warnings: list[str]) -> None:
    batch = row["batch"]
    lozo_config = row["lozo"].get("config", {})
    vllm_config = row["vllm"].get("config", {})
    if lozo_config.get("backend") != "lozo":
        errors.append(f"batch {batch}: LOZO backend is {lozo_config.get('backend')!r}")
    if vllm_config.get("backend") != "vllm":
        errors.append(f"batch {batch}: vLLM backend is {vllm_config.get('backend')!r}")
    if lozo_config.get("profile_mode") != "minimal":
        errors.append(f"batch {batch}: LOZO profile is {lozo_config.get('profile_mode')!r}")
    if vllm_config.get("profile_mode") != "minimal":
        errors.append(f"batch {batch}: vLLM profile is {vllm_config.get('profile_mode')!r}")

    for key in COMMON_CONFIG_KEYS:
        lozo_value = lozo_config.get(key)
        vllm_value = vllm_config.get(key)
        if lozo_value != vllm_value:
            errors.append(
                f"batch {batch}: config mismatch for {key}: "
                f"LOZO={lozo_value!r}, vLLM={vllm_value!r}"
            )
        if key not in lozo_config:
            warnings.append(f"batch {batch}: legacy LOZO result missing config key {key}")
        if key not in vllm_config:
            warnings.append(f"batch {batch}: legacy vLLM result missing config key {key}")
    if lozo_config.get("batch_size") != batch:
        errors.append(
            f"batch {batch}: LOZO batch_size={lozo_config.get('batch_size')!r}"
        )
    if vllm_config.get("batch_size") != batch:
        errors.append(
            f"batch {batch}: vLLM batch_size={vllm_config.get('batch_size')!r}"
        )

    for key, expected in VLLM_EXPECTED_CONFIG.items():
        actual = vllm_config.get(key)
        if actual != expected:
            errors.append(
                f"batch {batch}: unexpected vLLM {key}={actual!r}, expected {expected!r}"
            )


def check_metrics(row: dict, min_speedup: float | None, errors: list[str]) -> dict:
    batch = row["batch"]
    lozo = row["lozo"]
    vllm = row["vllm"]
    metrics = {
        "batch": batch,
        "lozo_total_s_per_step": total_per_step(lozo),
        "vllm_total_s_per_step": total_per_step(vllm),
        "lozo_raw_step_s_mean": step_mean(lozo),
        "vllm_raw_step_s_mean": step_mean(vllm),
        "lozo_loss_drop": loss_drop(lozo),
        "vllm_loss_drop": loss_drop(vllm),
    }
    vllm_step = metrics["vllm_total_s_per_step"]
    metrics["speedup"] = (
        metrics["lozo_total_s_per_step"] / vllm_step if vllm_step > 0.0 else 0.0
    )

    for name in [
        "lozo_total_s_per_step",
        "vllm_total_s_per_step",
        "lozo_raw_step_s_mean",
        "vllm_raw_step_s_mean",
    ]:
        if metrics[name] <= 0.0:
            errors.append(f"batch {batch}: non-positive {name}={metrics[name]}")
    for name in ["lozo_loss_drop", "vllm_loss_drop"]:
        if metrics[name] <= 0.0:
            errors.append(f"batch {batch}: non-positive {name}={metrics[name]}")
    if min_speedup is not None and metrics["speedup"] < min_speedup:
        errors.append(
            f"batch {batch}: speedup {metrics['speedup']:.4f} below "
            f"threshold {min_speedup:.4f}"
        )
    return metrics


def check_detailed(row: dict, errors: list[str]) -> dict | None:
    detailed = row["detailed"]
    if detailed is None:
        return None
    batch = row["batch"]
    config = detailed.get("config", {})
    if config.get("backend") != "vllm" or config.get("profile_mode") != "detailed":
        errors.append(f"batch {batch}: malformed detailed vLLM result")
    if config.get("batch_size") != batch:
        errors.append(f"batch {batch}: detailed batch_size={config.get('batch_size')!r}")

    timing = detailed.get("timing", {})
    raw_step = step_mean(detailed)
    metrics = {
        "batch": batch,
        "vllm_detailed_total_s_per_step": total_per_step(detailed),
        "vllm_detailed_raw_step_s_mean": raw_step,
    }
    for key in DETAILED_TIMING_KEYS:
        value = timing.get(key)
        if not isinstance(value, dict):
            errors.append(f"batch {batch}: missing detailed timing key {key}")
            continue
        mean = float(value.get("mean", 0.0))
        if mean <= 0.0 and key not in {"score_request_build_s", "score_postprocess_s"}:
            errors.append(f"batch {batch}: non-positive detailed {key} mean={mean}")
        metrics[key] = mean
    score_generate = metrics.get("score_generate_s", 0.0)
    metrics["score_generate_share"] = score_generate / raw_step if raw_step else 0.0
    return metrics


def validate(
    run_dir: Path,
    min_speedup: float | None,
    expected_batches: list[int],
    require_detailed_batches: list[int],
) -> dict:
    errors = []
    warnings = []
    if not run_dir.exists():
        raise SystemExit(f"run directory does not exist: {run_dir}")
    batch_dirs = discover_batch_dirs(run_dir)
    if not batch_dirs:
        raise SystemExit(f"no batch_b* directories found under: {run_dir}")

    rows = [row_for_batch(path) for path in batch_dirs]
    found_batches = [row["batch"] for row in rows]
    if expected_batches and found_batches != expected_batches:
        errors.append(
            f"batch set mismatch: found {found_batches}, expected {expected_batches}"
        )
    required_detailed = set(require_detailed_batches)
    speed = []
    detailed_timing = []
    artifacts = {}
    for row in rows:
        batch = row["batch"]
        artifacts[str(batch)] = {
            "lozo": display_path(row["lozo_path"]) if row["lozo_path"] else None,
            "vllm": display_path(row["vllm_path"]) if row["vllm_path"] else None,
            "detailed": display_path(row["detailed_path"])
            if row["detailed_path"]
            else None,
        }
        if batch in required_detailed and row["detailed"] is None:
            errors.append(f"batch {batch}: missing required detailed vLLM result")
        if not check_required(row, errors):
            continue
        check_configs(row, errors, warnings)
        speed.append(check_metrics(row, min_speedup, errors))
        detailed = check_detailed(row, errors)
        if detailed:
            detailed_timing.append(detailed)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "run_dir": display_path(run_dir),
        "batches": found_batches,
        "expected_batches": expected_batches,
        "require_detailed_batches": require_detailed_batches,
        "speed": speed,
        "detailed_timing": detailed_timing,
        "artifacts": artifacts,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a Phase 3 q=1 batch sweep.")
    parser.add_argument("run_dir", help="Batch sweep run directory.")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="Optional per-batch vLLM speedup threshold.",
    )
    parser.add_argument(
        "--expected-batches",
        default=None,
        help="Comma-separated batch sizes that must be present in order.",
    )
    parser.add_argument(
        "--require-detailed-batches",
        default=None,
        help="Comma-separated batch sizes that must include vLLM detailed results.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.run_dir = resolve_path(args.run_dir)
    args.expected_batches = parse_int_list(args.expected_batches)
    args.require_detailed_batches = parse_int_list(args.require_detailed_batches)
    return args


def main() -> None:
    args = parse_args()
    report = validate(
        args.run_dir,
        args.min_speedup,
        args.expected_batches,
        args.require_detailed_batches,
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
