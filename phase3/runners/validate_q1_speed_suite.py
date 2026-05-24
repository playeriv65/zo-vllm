import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phase3.runners.collect_q1_speed_suite import (  # noqa: E402
    dedupe_paths,
    display_path,
    discover_results,
    load_result,
    step_mean,
    total_per_step,
)


REQUIRED_RESULTS = [
    ("lozo", "minimal"),
    ("vllm", "minimal"),
    ("vllm", "detailed"),
]

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


def resolve_run_dirs(run_dir_args: list[str]) -> list[Path]:
    run_dirs = []
    for run_dir_arg in run_dir_args:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        run_dirs.append(run_dir)
    return run_dirs


def load_rows(run_dirs: list[Path]) -> dict[tuple[str, str], dict]:
    paths = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise SystemExit(f"run directory does not exist: {run_dir}")
        paths.extend(discover_results(run_dir))

    rows = {}
    for path in dedupe_paths(paths):
        data = load_result(path)
        config = data.get("config", {})
        key = (config.get("backend", "unknown"), config.get("profile_mode", "unknown"))
        rows[key] = {"path": path, "data": data}
    return rows


def check_common_config(
    rows: dict[tuple[str, str], dict],
    errors: list[str],
    warnings: list[str],
) -> None:
    reference = rows[("lozo", "minimal")]["data"].get("config", {})
    for key in COMMON_CONFIG_KEYS:
        expected = reference.get(key)
        for result_key in REQUIRED_RESULTS[1:]:
            actual = rows[result_key]["data"].get("config", {}).get(key)
            if actual != expected:
                errors.append(
                    f"config mismatch for {key}: {result_key} has {actual!r}, "
                    f"expected {expected!r}"
                )
        if key not in reference:
            warnings.append(f"legacy result missing config key: lozo minimal {key}")
        for result_key in REQUIRED_RESULTS[1:]:
            config = rows[result_key]["data"].get("config", {})
            if key not in config:
                warnings.append(f"legacy result missing config key: {result_key} {key}")


def check_vllm_defaults(rows: dict[tuple[str, str], dict], errors: list[str]) -> None:
    for result_key in [("vllm", "minimal"), ("vllm", "detailed")]:
        config = rows[result_key]["data"].get("config", {})
        expected = {
            "batch_invariant": "0",
            "enforce_eager": "0",
            "lora_residency": "gpu",
            "lora_injection_resolved": "direct",
            "weight_update": "direct",
            "weight_update_precision": "param",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                errors.append(
                    f"unexpected vLLM config for {result_key} {key}: "
                    f"{config.get(key)!r}, expected {value!r}"
                )


def loss_drop(data: dict) -> float:
    return -float(data.get("loss_change", 0.0))


def check_loss_and_speed(
    rows: dict[tuple[str, str], dict],
    min_vllm_speedup: float,
    errors: list[str],
) -> dict:
    lozo = rows[("lozo", "minimal")]["data"]
    vllm_minimal = rows[("vllm", "minimal")]["data"]
    vllm_detailed = rows[("vllm", "detailed")]["data"]

    for result_key, row in rows.items():
        drop = loss_drop(row["data"])
        if drop <= 0.0:
            errors.append(f"non-positive loss drop for {result_key}: {drop}")
        if total_per_step(row["data"]) <= 0.0:
            errors.append(f"non-positive total_s/step for {result_key}")
        if step_mean(row["data"]) <= 0.0:
            errors.append(f"non-positive raw step mean for {result_key}")

    lozo_step = total_per_step(lozo)
    vllm_minimal_step = total_per_step(vllm_minimal)
    vllm_detailed_step = total_per_step(vllm_detailed)
    minimal_speedup = lozo_step / vllm_minimal_step if vllm_minimal_step else 0.0
    detailed_speedup = lozo_step / vllm_detailed_step if vllm_detailed_step else 0.0
    if minimal_speedup < min_vllm_speedup:
        errors.append(
            f"vLLM minimal speedup {minimal_speedup:.4f} is below "
            f"threshold {min_vllm_speedup:.4f}"
        )

    return {
        "lozo_total_s_per_step": lozo_step,
        "vllm_minimal_total_s_per_step": vllm_minimal_step,
        "vllm_detailed_total_s_per_step": vllm_detailed_step,
        "vllm_minimal_speedup": minimal_speedup,
        "vllm_detailed_speedup": detailed_speedup,
        "lozo_loss_drop": loss_drop(lozo),
        "vllm_minimal_loss_drop": loss_drop(vllm_minimal),
        "vllm_detailed_loss_drop": loss_drop(vllm_detailed),
    }


def check_detailed_timing(rows: dict[tuple[str, str], dict], errors: list[str]) -> dict:
    detailed = rows[("vllm", "detailed")]["data"]
    timing = detailed.get("timing", {})
    raw_step = step_mean(detailed)
    metrics = {}
    for key in DETAILED_TIMING_KEYS:
        value = timing.get(key)
        if not isinstance(value, dict):
            errors.append(f"missing detailed timing key: {key}")
            continue
        mean = float(value.get("mean", 0.0))
        if mean <= 0.0 and key not in {"score_request_build_s", "score_postprocess_s"}:
            errors.append(f"non-positive detailed timing mean for {key}: {mean}")
        metrics[key] = mean

    score_generate = metrics.get("score_generate_s", 0.0)
    metrics["score_generate_share"] = score_generate / raw_step if raw_step else 0.0
    return metrics


def validate(run_dirs: list[Path], min_vllm_speedup: float) -> dict:
    errors = []
    warnings = []
    rows = load_rows(run_dirs)
    missing = [key for key in REQUIRED_RESULTS if key not in rows]
    for key in missing:
        errors.append(f"missing required result: {key}")

    if not missing:
        check_common_config(rows, errors, warnings)
        check_vllm_defaults(rows, errors)
        speed_metrics = check_loss_and_speed(rows, min_vllm_speedup, errors)
        detailed_metrics = check_detailed_timing(rows, errors)
    else:
        speed_metrics = {}
        detailed_metrics = {}

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "run_dirs": [display_path(path) for path in run_dirs],
        "artifacts": {
            f"{backend}_{profile}": display_path(row["path"])
            for (backend, profile), row in sorted(rows.items())
        },
        "speed": speed_metrics,
        "detailed_timing": detailed_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a Phase 3 q=1 speed suite.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories under phase3/results or absolute paths.",
    )
    parser.add_argument("--min-vllm-speedup", type=float, default=1.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.run_dirs = resolve_run_dirs(args.run_dirs)
    return args


def main() -> None:
    args = parse_args()
    report = validate(args.run_dirs, args.min_vllm_speedup)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text, flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
