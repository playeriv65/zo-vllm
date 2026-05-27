import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def newest_json(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def load_result(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def step_mean(data: dict) -> float:
    timing = data.get("timing", {})
    tail_100 = timing.get("tail_100", {})
    if isinstance(tail_100, dict) and isinstance(tail_100.get("step_s"), dict):
        return float(tail_100["step_s"].get("mean", 0.0))
    if isinstance(timing.get("step_s"), dict):
        return float(timing["step_s"].get("mean", 0.0))
    return float(timing.get("step_s_mean", 0.0))


def total_per_step(data: dict) -> float:
    steps = data.get("config", {}).get("steps")
    if not steps:
        return 0.0
    return float(data.get("timing", {}).get("total_s", 0.0)) / float(steps)


def loss_drop_per_second(loss_drop: float, total_s: float) -> float:
    if total_s <= 0:
        return 0.0
    return loss_drop / total_s


def fmt(value: float | int | str | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def discover_results(run_dir: Path) -> list[Path]:
    candidates = [
        newest_json(run_dir / "lozo_minimal", "lozo_perf_minimal_*.json"),
        newest_json(run_dir / "vllm_minimal_eager0", "vllm_perf_minimal_*.json"),
        newest_json(run_dir / "vllm_minimal_eager1", "vllm_perf_minimal_*.json"),
        newest_json(run_dir / "vllm_detailed_eager0", "vllm_perf_detailed_*.json"),
        newest_json(run_dir / "vllm_detailed_eager1", "vllm_perf_detailed_*.json"),
    ]
    return [path for path in candidates if path is not None]


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    deduped = []
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def row_order(row: dict) -> tuple[int, str]:
    order = {
        ("lozo", "minimal"): 0,
        ("lozo", "instrumented"): 1,
        ("vllm", "minimal"): 2,
        ("vllm", "detailed"): 3,
    }
    return (order.get((row["backend"], row["profile"]), 99), str(row["path"]))


def build_summary(run_dirs: list[Path], json_paths: list[Path]) -> str:
    rows = []
    for path in json_paths:
        data = load_result(path)
        config = data.get("config", {})
        loss_change = float(data.get("loss_change", 0.0))
        rows.append({
            "path": path,
            "backend": config.get("backend", "unknown"),
            "profile": config.get("profile_mode", "unknown"),
            "steps": config.get("steps"),
            "batch": config.get("batch_size"),
            "initial": data.get("initial_loss"),
            "final": data.get("final_loss"),
            "loss_drop": -loss_change,
            "total_s": float(data.get("timing", {}).get("total_s", 0.0)),
            "total_s_per_step": total_per_step(data),
            "raw_step_s_mean": step_mean(data),
            "timing": data.get("timing", {}),
        })

    rows.sort(key=row_order)
    baseline = rows[0] if rows else None
    lines = [
        "# Phase 3 q=1 Speed Suite Summary",
        "",
        "Run directories:",
        "",
    ]
    for run_dir in run_dirs:
        lines.append(f"- `{display_path(run_dir)}`")
    lines.extend([
        "",
        "| backend | profile | steps | batch | initial | final | loss_drop | loss_drop/s | total_s | total_s/step | raw_step_s_mean | speedup_vs_first |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        speedup = None
        if baseline and baseline["total_s_per_step"] and row["total_s_per_step"]:
            speedup = baseline["total_s_per_step"] / row["total_s_per_step"]
        lines.append(
            "| {backend} | {profile} | {steps} | {batch} | {initial} | {final} | "
            "{loss_drop} | {loss_drop_per_second} | {total_s} | {total_s_per_step} | "
            "{raw_step_s_mean} | {speedup} |".format(
                backend=row["backend"],
                profile=row["profile"],
                steps=row["steps"],
                batch=row["batch"],
                initial=fmt(row["initial"]),
                final=fmt(row["final"]),
                loss_drop=fmt(row["loss_drop"]),
                loss_drop_per_second=fmt(
                    loss_drop_per_second(row["loss_drop"], row["total_s"])
                ),
                total_s=fmt(row["total_s"], 4),
                total_s_per_step=fmt(row["total_s_per_step"]),
                raw_step_s_mean=fmt(row["raw_step_s_mean"]),
                speedup=fmt(speedup, 4),
            )
        )

    detailed = next(
        (row for row in rows if row["backend"] == "vllm" and row["profile"] == "detailed"),
        None,
    )
    if detailed:
        timing = detailed["timing"]
        lines.extend([
            "",
            "## vLLM Detailed Timing",
            "",
            "| component | mean s/step | share of raw step |",
            "|---|---:|---:|",
        ])
        for key in [
            "score_s",
            "score_generate_s",
            "score_request_build_s",
            "score_postprocess_s",
            "lora_update_s",
            "weight_update_s",
            "weight_fold_s",
            "build_lora_s",
            "direction_s",
        ]:
            value = timing.get(key, {})
            mean = value.get("mean", 0.0) if isinstance(value, dict) else 0.0
            share = float(mean) / detailed["raw_step_s_mean"] if detailed["raw_step_s_mean"] else 0.0
            lines.append(f"| {key} | {fmt(float(mean))} | {fmt(share * 100.0, 2)}% |")

    lines.extend([
        "",
        "## Artifacts",
        "",
    ])
    for row in rows:
        lines.append(f"- `{display_path(row['path'])}`")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Phase 3 q=1 speed suite runs.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories under phase3/results or absolute paths.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dirs = []
    for run_dir_arg in args.run_dirs:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        run_dirs.append(run_dir)
    args.run_dirs = run_dirs
    return args


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dirs:
        if not run_dir.exists():
            raise SystemExit(f"run directory does not exist: {run_dir}")

    json_paths = []
    for run_dir in args.run_dirs:
        json_paths.extend(discover_results(run_dir))
    json_paths = dedupe_paths(json_paths)
    if not json_paths:
        run_dir_text = ", ".join(str(path) for path in args.run_dirs)
        raise SystemExit(f"no result JSON files found under: {run_dir_text}")

    if args.output:
        output = Path(args.output)
    elif len(args.run_dirs) == 1:
        output = args.run_dirs[0] / "summary.md"
    else:
        output = args.run_dirs[0] / "summary_combined.md"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    summary = build_summary(args.run_dirs, json_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary)
    print(summary)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
