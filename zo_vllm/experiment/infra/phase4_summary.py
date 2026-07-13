from pathlib import Path
import re

from zo_vllm.utils.io import load_json
from .run_state import read_run_state


def _command_flag_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    idx = command.index(flag)
    if idx + 1 >= len(command):
        return None
    return str(command[idx + 1])


def _extract_wandb_url_from_log(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    url_pattern = re.compile(r"https://wandb\.ai/\S+")
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None
    matches = url_pattern.findall(text)
    return matches[-1] if matches else None


def summarize_phase4_job(job_dir: Path) -> dict:
    result_path = job_dir / "result.json"
    manifest_path = job_dir / "manifest.json"
    state = read_run_state(job_dir)

    row = {
        "job_id": job_dir.name,
        "status": state.get("status"),
        "resume_count": state.get("resume_count", 0),
        "attempts": state.get("attempts", 0),
        "converged": None,
        "backend": None,
        "train_scope": None,
        "final_metric": None,
        "initial_loss": None,
        "final_loss": None,
        "wall_clock_s": None,
        "step_time_s": None,
        "steps_per_sec": None,
        "wandb_url": None,
        "log_file": str(job_dir / "logs" / "run.log"),
        "artifact_json": None,
        "model_name": None,
        "task_name": None,
        "steps": None,
        "batch_size": None,
        "lr": None,
        "eps": None,
        "rank": None,
    }

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        row["backend"] = manifest.get("backend")
        row["train_scope"] = manifest.get("train_scope")
        row["model_name"] = manifest.get("model_name", row["model_name"])
        row["task_name"] = manifest.get("task_name", row["task_name"])
        row["steps"] = manifest.get("steps", row["steps"])
        row["batch_size"] = manifest.get("batch_size", row["batch_size"])
        row["lr"] = manifest.get("lr", row["lr"])
        row["eps"] = manifest.get("eps", row["eps"])
        row["rank"] = manifest.get("rank", row["rank"])
        command = manifest.get("command", [])
        if isinstance(command, list):
            model_name = _command_flag_value(command, "--model-name")
            task_name = _command_flag_value(command, "--task-name")
            steps = _command_flag_value(command, "--steps")
            batch_size = _command_flag_value(command, "--batch-size")
            lr = _command_flag_value(command, "--lr")
            eps = _command_flag_value(command, "--eps")
            rank = _command_flag_value(command, "--rank")
            if model_name is not None:
                row["model_name"] = model_name
            if task_name is not None:
                row["task_name"] = task_name
            if steps is not None:
                row["steps"] = steps
            if batch_size is not None:
                row["batch_size"] = batch_size
            if lr is not None:
                row["lr"] = lr
            if eps is not None:
                row["eps"] = eps
            if rank is not None:
                row["rank"] = rank

    if result_path.exists():
        data = load_json(result_path)
        row.update(
            {
                "backend": data.get("backend", row["backend"]),
                "train_scope": data.get("config", {}).get("train_scope", row["train_scope"]),
                "converged": data.get("converged"),
                "initial_loss": data.get("initial_loss"),
                "final_loss": data.get("final_loss"),
                "final_metric": data.get("final_loss"),
                "final_accuracy": data.get("final_accuracy"),
                "wall_clock_s": data.get("wall_clock_s"),
                "step_time_s": data.get("step_time_s"),
                "steps_per_sec": data.get("steps_per_sec"),
                "wandb_url": data.get("wandb_url"),
                "log_file": data.get("log_file", row["log_file"]),
                "artifact_json": data.get("artifact_json"),
            }
        )
    if row["wandb_url"] is None:
        row["wandb_url"] = _extract_wandb_url_from_log(Path(row["log_file"]))
    if "final_accuracy" not in row:
        row["final_accuracy"] = None
    return row
