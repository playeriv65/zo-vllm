import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.paths import resolve_path
from zo_vllm.experiment.run_state import read_run_state


def display(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def collect(run_dir: Path) -> dict:
    jobs_dir = run_dir / "jobs"
    rows = []
    if jobs_dir.exists():
        for job_dir in sorted(path for path in jobs_dir.iterdir() if path.is_dir()):
            state = read_run_state(job_dir)
            rows.append(
                {
                    "job_id": job_dir.name,
                    "status": state.get("status"),
                    "attempts": state.get("attempts", 0),
                    "resume_count": state.get("resume_count", 0),
                    "has_result": (job_dir / "phase4_result.json").exists(),
                    "has_manifest": (job_dir / "manifest.json").exists(),
                    "has_log": (job_dir / "logs" / "run.log").exists(),
                }
            )
    return {
        "run_dir": display(run_dir),
        "manifest_exists": (run_dir / "manifest.json").exists(),
        "summary_exists": (run_dir / "summary.md").exists(),
        "jobs": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    report = collect(run_dir)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
