import importlib.util
from pathlib import Path


LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1] / "phase4" / "runners" / "launch_phase4.py"
)
SPEC = importlib.util.spec_from_file_location("phase4_launch_phase4", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launch_phase4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch_phase4)

build_job_cmd = launch_phase4.build_job_cmd
find_latest_lora_checkpoint = launch_phase4.find_latest_lora_checkpoint


def test_phase4_resume_existing_uses_latest_lora_checkpoint(tmp_path):
    run_dir = tmp_path / "phase4_run"
    job_id = "job-a"
    checkpoint_root = run_dir / "jobs" / job_id / "artifacts" / "checkpoints"
    older = checkpoint_root / "checkpoint-0000010"
    latest = checkpoint_root / "checkpoint-0000020"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)
    (older / "zo_lora_bank.pt").write_bytes(b"old")
    (latest / "zo_lora_bank.pt").write_bytes(b"new")

    assert find_latest_lora_checkpoint(run_dir / "jobs" / job_id) == latest

    defaults = {
        "backend": "vllm",
        "model_name": "facebook/opt-125m",
        "steps": 20,
        "warmup_steps": 0,
        "batch_size": 2,
        "eval_interval": 10,
        "seed": 1,
        "rank": 2,
        "lr": 1e-7,
        "eps": 1e-3,
        "nu": -1,
        "save_strategy": "steps",
        "save_steps": 10,
        "save_total_limit": 2,
        "save_checkpoint_mode": "lora",
        "load_best_model_at_end": 0,
        "save_final_checkpoint": 0,
        "task": {
            "name": "sst2",
            "num_train": 16,
            "num_dev": 8,
            "num_eval": 8,
            "data_seed": 1,
        },
    }
    job = {"job_id": job_id}

    cmd = build_job_cmd(job, defaults, {}, run_dir, "0", resumed=True)

    checkpoint_arg_index = cmd.index("--resume-lora-checkpoint") + 1
    assert cmd[checkpoint_arg_index] == str(latest)
    assert "--resume" in cmd
