import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest does not exist: {manifest_path}")
    with manifest_path.open() as f:
        return json.load(f)


def run_command(command: list[str], dry_run: bool) -> None:
    print("command=" + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run collect and validate commands from a Phase 3 manifest."
    )
    parser.add_argument("run_dir", help="Run directory containing manifest.json.")
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Only run the manifest collect command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()
    args.run_dir = resolve_path(args.run_dir)
    return args


def main() -> None:
    args = parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run directory does not exist: {args.run_dir}")

    manifest = load_manifest(args.run_dir)
    collect_command = manifest.get("collect_command")
    validate_command = manifest.get("validate_command")
    if not isinstance(collect_command, list) or not collect_command:
        raise SystemExit("manifest is missing collect_command")
    if (
        not args.skip_validate
        and validate_command is not None
        and not isinstance(validate_command, list)
    ):
        raise SystemExit("manifest validate_command must be a list or null")

    run_command([str(item) for item in collect_command], args.dry_run)
    if not args.skip_validate and validate_command:
        run_command([str(item) for item in validate_command], args.dry_run)
    print("finalize_ok=true", flush=True)


if __name__ == "__main__":
    main()
