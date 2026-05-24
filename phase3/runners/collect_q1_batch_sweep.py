import argparse
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


def discover_batch_dirs(run_dir: Path) -> list[Path]:
    dirs = []
    for path in run_dir.glob("batch_b*"):
        if path.is_dir():
            dirs.append(path)
    return sorted(dirs, key=batch_sort_key)


def batch_sort_key(path: Path) -> int:
    name = path.name
    if name.startswith("batch_b"):
        try:
            return int(name.removeprefix("batch_b"))
        except ValueError:
            return 10**9
    return 10**9


def load_optional_result(directory: Path, pattern: str) -> tuple[Path | None, dict | None]:
    path = newest_json(directory, pattern)
    if path is None:
        return None, None
    return path, load_result(path)


def loss_drop(data: dict | None) -> float | None:
    if data is None:
        return None
    return -float(data.get("loss_change", 0.0))


def row_for_batch(batch_dir: Path) -> dict:
    lozo_path, lozo = load_optional_result(
        batch_dir / "lozo_minimal", "lozo_perf_minimal_*.json"
    )
    vllm_path, vllm = load_optional_result(
        batch_dir / "vllm_minimal_eager0", "vllm_perf_minimal_*.json"
    )
    detailed_path, detailed = load_optional_result(
        batch_dir / "vllm_detailed_eager0", "vllm_perf_detailed_*.json"
    )

    batch_size = batch_sort_key(batch_dir)
    lozo_step = total_per_step(lozo) if lozo else None
    vllm_step = total_per_step(vllm) if vllm else None
    detailed_step = total_per_step(detailed) if detailed else None
    speedup = lozo_step / vllm_step if lozo_step and vllm_step else None
    detailed_speedup = lozo_step / detailed_step if lozo_step and detailed_step else None

    return {
        "batch": batch_size,
        "batch_dir": batch_dir,
        "lozo_path": lozo_path,
        "vllm_path": vllm_path,
        "detailed_path": detailed_path,
        "lozo": lozo,
        "vllm": vllm,
        "detailed": detailed,
        "lozo_step": lozo_step,
        "vllm_step": vllm_step,
        "detailed_step": detailed_step,
        "speedup": speedup,
        "detailed_speedup": detailed_speedup,
    }


def build_summary(run_dir: Path, rows: list[dict]) -> str:
    lines = [
        "# Phase 3 q=1 Batch Sweep Summary",
        "",
        f"Run directory: `{display_path(run_dir)}`",
        "",
        "| batch | lozo_loss_drop | vllm_loss_drop | lozo_total_s/step | vllm_total_s/step | speedup | vllm_detailed_s/step | detailed_speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {batch} | {lozo_drop} | {vllm_drop} | {lozo_step} | {vllm_step} | "
            "{speedup} | {detailed_step} | {detailed_speedup} |".format(
                batch=row["batch"],
                lozo_drop=fmt(loss_drop(row["lozo"])),
                vllm_drop=fmt(loss_drop(row["vllm"])),
                lozo_step=fmt(row["lozo_step"]),
                vllm_step=fmt(row["vllm_step"]),
                speedup=fmt(row["speedup"], 4),
                detailed_step=fmt(row["detailed_step"]),
                detailed_speedup=fmt(row["detailed_speedup"], 4),
            )
        )

    detailed_rows = [row for row in rows if row["detailed"]]
    if detailed_rows:
        lines.extend([
            "",
            "## Detailed Timing",
            "",
            "| batch | score_generate_s | score_generate_share | lora_update_s | weight_update_s | build_lora_s | direction_s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in detailed_rows:
            timing = row["detailed"].get("timing", {})
            raw_step = step_mean(row["detailed"])
            score_generate = float(timing.get("score_generate_s", {}).get("mean", 0.0))
            score_share = score_generate / raw_step if raw_step else 0.0
            lines.append(
                "| {batch} | {score_generate} | {score_share} | {lora_update} | "
                "{weight_update} | {build_lora} | {direction} |".format(
                    batch=row["batch"],
                    score_generate=fmt(score_generate),
                    score_share=f"{score_share * 100.0:.2f}%",
                    lora_update=fmt(float(timing.get("lora_update_s", {}).get("mean", 0.0))),
                    weight_update=fmt(float(timing.get("weight_update_s", {}).get("mean", 0.0))),
                    build_lora=fmt(float(timing.get("build_lora_s", {}).get("mean", 0.0))),
                    direction=fmt(float(timing.get("direction_s", {}).get("mean", 0.0))),
                )
            )

    lines.extend(["", "## Artifacts", ""])
    for row in rows:
        for key in ["lozo_path", "vllm_path", "detailed_path"]:
            path = row[key]
            if path is not None:
                lines.append(f"- `{display_path(path)}`")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Collect a Phase 3 q=1 batch sweep.")
    parser.add_argument("run_dir", help="Batch sweep run directory.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.run_dir = resolve_path(args.run_dir)
    return args


def main() -> None:
    args = parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run directory does not exist: {args.run_dir}")

    batch_dirs = discover_batch_dirs(args.run_dir)
    if not batch_dirs:
        raise SystemExit(f"no batch_b* directories found under: {args.run_dir}")
    rows = [row_for_batch(path) for path in batch_dirs]
    summary = build_summary(args.run_dir, rows)

    output = Path(args.output) if args.output else args.run_dir / "summary.md"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary)
    print(summary)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
