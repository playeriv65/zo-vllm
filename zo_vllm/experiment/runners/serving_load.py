"""Common serving-load experiment runner for serving-time ZO training."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from zo_vllm.config import DEFAULT_ZO_SERVING_CONFIG
from zo_vllm.core.lora_scope import (
    resolve_lora_target_modules,
    resolve_update_bank_rank,
)
from zo_vllm.training.model_metadata import (
    build_lora_param_metadata_from_config,
    estimate_lora_bank_state_memory_bytes,
    resolve_direction_dtype,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVING_DEFAULT = DEFAULT_ZO_SERVING_CONFIG


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: str = "0") -> bool:
    return _env(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_optional(name: str) -> str | None:
    value = os.environ.get(name)
    return None if value is None or not value.strip() else value


def _json_request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> str:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _stream_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return int(process.wait())


def _run_logged_python_script(args: list[str], *, log_path: Path) -> int:
    return _stream_command(
        [str(_python_executable()), *args],
        env=_runtime_env(),
        log_path=log_path,
    )


def _python_executable() -> Path:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    gpu = _env("GPU", _env("CUDA_VISIBLE_DEVICES"))
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VLLM_ZO_SERVING_TRAINING"] = "1"
    env["VLLM_ZO_FORCE_EAGER_SCORING"] = _env("VLLM_ZO_FORCE_EAGER_SCORING", "1")
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _format_gib(value: int) -> str:
    return f"{float(value) / (1 << 30):.4f}"


class ServingLoadConfig:
    """Environment-backed configuration for serving load experiments."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.gpu = _env("GPU", _env("CUDA_VISIBLE_DEVICES"))
        self.host = _env("HOST", "127.0.0.1")
        self.port = _env("PORT", "8007")
        self.base_url = _env("BASE_URL", f"http://{self.host}:{self.port}")
        self.model = _env("MODEL", "Qwen/Qwen3-8B")
        self.task_name = _env("ZO_TASK_NAME", SERVING_DEFAULT.task_name)
        self.served_model_name = _env("SERVED_MODEL_NAME", self.model)
        self.bench_model = _env("BENCH_MODEL", self.model)
        self.bench_tokenizer = _env("BENCH_TOKENIZER", self.model)
        self.dtype = _env("DTYPE", "auto")
        self.vllm_quantization = _env("VLLM_QUANTIZATION")
        self.compilation_config = _env("COMPILATION_CONFIG")
        self.async_scheduling = _env("ASYNC_SCHEDULING")
        self.max_model_len = _env("MAX_MODEL_LEN", "2048")
        self.max_num_batched_tokens = _env("MAX_NUM_BATCHED_TOKENS")
        self.max_num_seqs = _env("MAX_NUM_SEQS")
        self.enable_prefix_caching = _env_optional("ENABLE_PREFIX_CACHING")
        self.gpu_memory_utilization = _env("GPU_MEMORY_UTILIZATION", "0.82")
        self.update_bank_rank_requested = _env(
            "UPDATE_BANK_RANK",
            str(SERVING_DEFAULT.update_bank_rank),
        )
        self.include_lm_head = _env_bool(
            "ZO_INCLUDE_LM_HEAD",
            "1" if SERVING_DEFAULT.include_lm_head else "0",
        )
        self.include_embeddings = _env_bool(
            "ZO_INCLUDE_EMBEDDINGS",
            "1" if SERVING_DEFAULT.include_embeddings else "0",
        )
        self.zo_target_modules = _env("ZO_TARGET_MODULES", _env("LORA_TARGET_MODULES"))
        self.lora_target_modules = resolve_lora_target_modules(
            self.zo_target_modules,
            include_lm_head=self.include_lm_head,
            include_embeddings=self.include_embeddings,
        )
        self.rank = _env("RANK", str(SERVING_DEFAULT.rank))
        self.eps = _env("EPS", str(SERVING_DEFAULT.eps))
        self.lr = _env("LR", str(SERVING_DEFAULT.learning_rate))
        self.weight_decay = _env("WEIGHT_DECAY", str(SERVING_DEFAULT.weight_decay))
        self.lr_scheduler_type = _env(
            "LR_SCHEDULER_TYPE",
            SERVING_DEFAULT.lr_scheduler_type,
        )
        self.warmup_steps = _env("WARMUP_STEPS", str(SERVING_DEFAULT.warmup_steps))
        self.nu = _env("NU", str(SERVING_DEFAULT.nu))
        self.seed = _env("SEED", str(SERVING_DEFAULT.seed))
        self.data_seed = _env_optional("DATA_SEED")
        self.batch_size = _env("BATCH_SIZE", str(SERVING_DEFAULT.batch_size))
        self.train_steps = _env("TRAIN_STEPS", str(SERVING_DEFAULT.steps))
        self.update_bank_rank, self.update_bank_rank_auto = resolve_update_bank_rank(
            self.update_bank_rank_requested,
            rank=int(self.rank),
            steps=int(self.train_steps),
            nu=int(self.nu),
        )
        self.update_bank_rank = str(self.update_bank_rank)
        self.plus_id = _env("PLUS_ID", str(SERVING_DEFAULT.slot.plus_id))
        self.minus_id = _env("MINUS_ID", str(SERVING_DEFAULT.slot.minus_id))
        self.num_train = _env("NUM_TRAIN", str(SERVING_DEFAULT.num_train))
        self.num_dev = _env("NUM_DEV", str(SERVING_DEFAULT.num_dev))
        self.eval_interval = _env("EVAL_INTERVAL", str(SERVING_DEFAULT.eval_interval))
        self.gradient_accumulation_update_steps = _env(
            "GRADIENT_ACCUMULATION_UPDATE_STEPS",
            str(SERVING_DEFAULT.gradient_accumulation_update_steps),
        )
        self.u_beta = _env("U_BETA", str(SERVING_DEFAULT.u_beta))
        self.u_norm_cap = _env_optional("U_NORM_CAP")
        self.random_device = _env("ZO_RANDOM_DEVICE", SERVING_DEFAULT.random_device)
        self.direction_device = _env(
            "DIRECTION_DEVICE",
            SERVING_DEFAULT.direction_device,
        )
        self.direction_sampling = _env(
            "DIRECTION_SAMPLING",
            SERVING_DEFAULT.direction_sampling,
        )
        self.direction_scale = _env(
            "DIRECTION_SCALE",
            str(SERVING_DEFAULT.direction_scale),
        )
        self.perturbation_normalization = _env(
            "PERTURBATION_NORMALIZATION",
            SERVING_DEFAULT.perturbation_normalization,
        )
        self.v_normalization = _env(
            "V_NORMALIZATION",
            SERVING_DEFAULT.v_normalization,
        )
        self.inter_step_delay_s = _env(
            "INTER_STEP_DELAY_S",
            str(SERVING_DEFAULT.inter_step_delay_s),
        )
        self.slot_write_stream = _env(
            "SLOT_WRITE_STREAM",
            SERVING_DEFAULT.slot_write_stream,
        )
        self.zo_admission_policy = _env(
            "ZO_ADMISSION_POLICY",
            SERVING_DEFAULT.score_admission_policy,
        )
        self.zo_admission_poll_s = _env(
            "ZO_ADMISSION_POLL_S",
            str(SERVING_DEFAULT.score_admission_poll_s),
        )
        self.zo_admission_timeout_s = _env(
            "ZO_ADMISSION_TIMEOUT_S",
            str(SERVING_DEFAULT.score_admission_timeout_s),
        )
        self.zo_admission_max_foreground_load = _env(
            "ZO_ADMISSION_MAX_FOREGROUND_LOAD",
            str(SERVING_DEFAULT.max_score_admission_foreground_load),
        )
        self.zo_admission_max_gpu_utilization = _env(
            "ZO_ADMISSION_MAX_GPU_UTILIZATION",
            str(SERVING_DEFAULT.max_score_admission_gpu_utilization),
        )
        self.zo_admission_gpu_device = _env(
            "ZO_ADMISSION_GPU_DEVICE",
            SERVING_DEFAULT.score_admission_gpu_device or self.gpu,
        )
        self.zo_queue_token_rate = _env(
            "ZO_QUEUE_TOKEN_RATE",
            str(SERVING_DEFAULT.zo_queue_token_rate),
        )
        self.zo_queue_burst_tokens = _env(
            "ZO_QUEUE_BURST_TOKENS",
            str(SERVING_DEFAULT.zo_queue_burst_tokens),
        )
        self.zo_queue_max_inflight = _env(
            "ZO_QUEUE_MAX_INFLIGHT",
            str(SERVING_DEFAULT.zo_queue_max_inflight),
        )
        self.zo_queue_max_admitted_tokens = _env(
            "ZO_QUEUE_MAX_ADMITTED_TOKENS",
            str(SERVING_DEFAULT.zo_queue_max_admitted_tokens),
        )
        self.zo_queue_poll_s = _env(
            "ZO_QUEUE_POLL_S",
            str(SERVING_DEFAULT.zo_queue_poll_s),
        )
        self.force_eager_scoring = _env("VLLM_ZO_FORCE_EAGER_SCORING", "1")
        self.enforce_eager = _env("ENFORCE_EAGER", "0")
        self.direction_dtype = _env("DIRECTION_DTYPE", SERVING_DEFAULT.direction_dtype)
        self.initial_eval = _env(
            "INITIAL_EVAL",
            "1" if SERVING_DEFAULT.initial_eval else "0",
        )
        self.enable_wandb = _env(
            "ENABLE_WANDB",
            "1" if SERVING_DEFAULT.enable_wandb else "0",
        )
        self.wandb_project = _env("WANDB_PROJECT", SERVING_DEFAULT.wandb_project)
        self.wandb_entity = _env("WANDB_ENTITY", SERVING_DEFAULT.wandb_entity)
        self.trust_remote_code = _env("TRUST_REMOTE_CODE", "0")
        self.zo_reserved_lora_bank_bytes = self._resolve_reserved_lora_bank_bytes()
        self.request_rate = _env("REQUEST_RATE", "1.0")
        self.burstiness = _env("BURSTINESS", "1.0")
        self.num_prompts = _env("NUM_PROMPTS", "1000")
        self.bench_seed = _env("BENCH_SEED", self.seed)
        self.dataset_name = _env("DATASET_NAME", "hf")
        if "DATASET_PATH" in os.environ:
            self.dataset_path = os.environ["DATASET_PATH"]
        elif self.dataset_name == "random":
            self.dataset_path = ""
        else:
            self.dataset_path = "Aeala/ShareGPT_Vicuna_unfiltered"
        self.hf_split = _env("HF_SPLIT", "train")
        self.hf_output_len = _env("HF_OUTPUT_LEN")
        self.input_len = _env("INPUT_LEN")
        self.output_len = _env("OUTPUT_LEN")
        self.temperature = _env("TEMPERATURE")
        self.goodput_ttft_ms = _env("GOODPUT_TTFT_MS", "200")
        self.goodput_tpot_ms = _env("GOODPUT_TPOT_MS", "50")
        self.goodput_e2el_ms = _env("GOODPUT_E2EL_MS", "10000")
        self.run_tag = _env("RUN_TAG", time.strftime("%Y%m%d_%H%M%S"))
        self.zo_output_dir = _env(
            "ZO_OUTPUT_DIR",
            str(PROJECT_ROOT / "phase7" / "logs" / "serving"),
        )
        self.log_root = Path(
            _env(
                "LOG_ROOT",
                str(PROJECT_ROOT / "phase7" / "logs" / "serving" / self.run_tag),
            )
        )
        self.sched_trace = _env_bool("SCHED_TRACE", "0")

    @property
    def params_path(self) -> Path:
        return self.log_root / f"params.{self.mode}.txt"

    def require_gpu(self) -> None:
        if not self.gpu:
            raise SystemExit(
                "[serving-runner] GPU is required for server mode; set GPU=<id> "
                "or CUDA_VISIBLE_DEVICES=<id>."
            )

    def server_cmd(self) -> list[str]:
        cmd = [
            str(_python_executable()),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            self.port,
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            self.gpu_memory_utilization,
            "--scheduling-policy",
            "priority",
            "--enable-server-load-tracking",
            "--enable-lora",
            "--max-loras",
            "2",
            "--max-lora-rank",
            self.update_bank_rank,
        ]
        if self.max_model_len:
            cmd.extend(["--max-model-len", self.max_model_len])
        if self.max_num_batched_tokens:
            cmd.extend(
                ["--max-num-batched-tokens", self.max_num_batched_tokens]
            )
        if self.max_num_seqs:
            cmd.extend(["--max-num-seqs", self.max_num_seqs])
        if self.enable_prefix_caching == "1":
            cmd.append("--enable-prefix-caching")
        elif self.enable_prefix_caching == "0":
            cmd.append("--no-enable-prefix-caching")
        if self.vllm_quantization:
            cmd.extend(["--quantization", self.vllm_quantization])
        if self.lora_target_modules:
            cmd.extend(["--lora-target-modules", *self.lora_target_modules])
        if self.compilation_config:
            cmd.extend(["--compilation-config", self.compilation_config])
        if self.async_scheduling == "1":
            cmd.append("--async-scheduling")
        elif self.async_scheduling == "0":
            cmd.append("--no-async-scheduling")
        if self.trust_remote_code == "1":
            cmd.append("--trust-remote-code")
        if self.enforce_eager == "1":
            cmd.append("--enforce-eager")
        return cmd

    def server_env(self) -> dict[str, str]:
        env = _runtime_env()
        env.setdefault(
            "VLLM_ZO_RESERVED_LORA_BANK_BYTES",
            str(int(self.zo_reserved_lora_bank_bytes)),
        )
        if self.sched_trace:
            env.setdefault(
                "VLLM_ZO_SCHED_TRACE_PATH",
                str(self.log_root / "scheduler_trace.jsonl"),
            )
        return env

    def bench_cmd(self, label: str) -> list[str]:
        cmd = [
            str(_python_executable()),
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--base-url",
            self.base_url,
            "--endpoint",
            "/v1/completions",
            "--model",
            self.bench_model,
        ]
        if self.served_model_name:
            cmd.extend(["--served-model-name", self.served_model_name])
        if self.bench_tokenizer:
            cmd.extend(["--tokenizer", self.bench_tokenizer])
        cmd.extend(["--dataset-name", self.dataset_name])
        if self.dataset_path and self.dataset_name != "random":
            cmd.extend(["--dataset-path", self.dataset_path])
        cmd.extend(
            [
                "--num-prompts",
                self.num_prompts,
                "--seed",
                self.bench_seed,
                "--request-rate",
                self.request_rate,
                "--burstiness",
                self.burstiness,
                "--save-result",
                "--save-detailed",
                "--plot-timeline",
                "--result-dir",
                str(self.log_root),
                "--result-filename",
                f"{label}_rps{self.request_rate}.json",
                "--goodput",
                f"ttft:{self.goodput_ttft_ms}",
                f"tpot:{self.goodput_tpot_ms}",
                f"e2el:{self.goodput_e2el_ms}",
            ]
        )
        if self.hf_split:
            cmd.extend(["--hf-split", self.hf_split])
        if self.hf_output_len:
            cmd.extend(["--hf-output-len", self.hf_output_len])
        if self.input_len:
            cmd.extend(["--input-len", self.input_len])
        if self.output_len:
            cmd.extend(["--output-len", self.output_len])
        if self.temperature:
            cmd.extend(["--temperature", self.temperature])
        return cmd

    def write_params(self) -> None:
        self.log_root.mkdir(parents=True, exist_ok=True)
        rows = {
            "mode": self.mode,
            "gpu": self.gpu,
            "model": self.model,
            "task_name": self.task_name,
            "served_model_name": self.served_model_name,
            "bench_model": self.bench_model,
            "bench_tokenizer": self.bench_tokenizer,
            "base_url": self.base_url,
            "dtype": self.dtype,
            "vllm_quantization": self.vllm_quantization,
            "compilation_config": self.compilation_config,
            "async_scheduling": self.async_scheduling,
            "cuda_launch_blocking": _env("CUDA_LAUNCH_BLOCKING"),
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "update_bank_rank": self.update_bank_rank,
            "update_bank_rank_requested": self.update_bank_rank_requested,
            "update_bank_rank_auto": self.update_bank_rank_auto,
            "zo_reserved_lora_bank_bytes": self.zo_reserved_lora_bank_bytes,
            "zo_reserved_lora_bank_gib": _format_gib(
                int(self.zo_reserved_lora_bank_bytes)
            ),
            "lora_target_modules": self.lora_target_modules,
            "zo_target_modules": self.zo_target_modules,
            "zo_include_lm_head": self.include_lm_head,
            "zo_include_embeddings": self.include_embeddings,
            "rank": self.rank,
            "eps": self.eps,
            "lr": self.lr,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "nu": self.nu,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "batch_size": self.batch_size,
            "train_steps": self.train_steps,
            "plus_id": self.plus_id,
            "minus_id": self.minus_id,
            "num_train": self.num_train,
            "num_dev": self.num_dev,
            "eval_interval": self.eval_interval,
            "gradient_accumulation_update_steps": (
                self.gradient_accumulation_update_steps
            ),
            "u_beta": self.u_beta,
            "u_norm_cap": self.u_norm_cap,
            "zo_random_device": self.random_device,
            "direction_device": self.direction_device,
            "direction_sampling": self.direction_sampling,
            "direction_scale": self.direction_scale,
            "perturbation_normalization": self.perturbation_normalization,
            "v_normalization": self.v_normalization,
            "inter_step_delay_s": self.inter_step_delay_s,
            "direction_dtype": self.direction_dtype,
            "initial_eval": self.initial_eval,
            "enable_wandb": self.enable_wandb,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "request_rate": self.request_rate,
            "burstiness": self.burstiness,
            "num_prompts": self.num_prompts,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "hf_split": self.hf_split,
            "hf_output_len": self.hf_output_len,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "temperature": self.temperature,
            "goodput_ttft_ms": self.goodput_ttft_ms,
            "goodput_tpot_ms": self.goodput_tpot_ms,
            "goodput_e2el_ms": self.goodput_e2el_ms,
            "sched_trace": self.sched_trace,
            "slot_write_stream": self.slot_write_stream,
            "zo_admission_policy": self.zo_admission_policy,
            "zo_admission_poll_s": self.zo_admission_poll_s,
            "zo_admission_timeout_s": self.zo_admission_timeout_s,
            "zo_admission_max_foreground_load": self.zo_admission_max_foreground_load,
            "zo_admission_max_gpu_utilization": (self.zo_admission_max_gpu_utilization),
            "zo_admission_gpu_device": self.zo_admission_gpu_device,
            "zo_queue_token_rate": self.zo_queue_token_rate,
            "zo_queue_burst_tokens": self.zo_queue_burst_tokens,
            "zo_queue_max_inflight": self.zo_queue_max_inflight,
            "zo_queue_max_admitted_tokens": self.zo_queue_max_admitted_tokens,
            "zo_queue_poll_s": self.zo_queue_poll_s,
            "vllm_zo_force_eager_scoring": self.force_eager_scoring,
            "enforce_eager": self.enforce_eager,
            "zo_output_dir": self.zo_output_dir,
        }
        text = "".join(f"{key}={value}\n" for key, value in rows.items())
        print(text, end="")
        self.params_path.write_text(text, encoding="utf-8")

    def zo_payload(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "run_name": self.run_tag,
            "output_dir": self.zo_output_dir,
            "steps": int(self.train_steps),
            "batch_size": int(self.batch_size),
            "plus_id": int(self.plus_id),
            "minus_id": int(self.minus_id),
            "num_train": int(self.num_train),
            "num_dev": int(self.num_dev),
            "eval_interval": int(self.eval_interval),
            "rank": int(self.rank),
            "update_bank_rank": int(self.update_bank_rank),
            "eps": float(self.eps),
            "learning_rate": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_steps": int(self.warmup_steps),
            "nu": int(self.nu),
            "seed": int(self.seed),
            "data_seed": None if self.data_seed is None else int(self.data_seed),
            "priority": int(_env("ZO_PRIORITY", str(SERVING_DEFAULT.priority))),
            "gradient_accumulation_update_steps": int(
                self.gradient_accumulation_update_steps
            ),
            "u_beta": float(self.u_beta),
            "u_norm_cap": None if self.u_norm_cap is None else float(self.u_norm_cap),
            "random_device": self.random_device,
            "direction_device": self.direction_device,
            "direction_sampling": self.direction_sampling,
            "direction_scale": (
                None if self.direction_scale is None else float(self.direction_scale)
            ),
            "perturbation_normalization": self.perturbation_normalization,
            "v_normalization": self.v_normalization,
            "target_modules": self.lora_target_modules,
            "include_lm_head": self.include_lm_head,
            "include_embeddings": self.include_embeddings,
            "direction_dtype": self.direction_dtype,
            "slot_write_stream": self.slot_write_stream,
            "score_admission_policy": self.zo_admission_policy,
            "score_admission_poll_s": float(self.zo_admission_poll_s),
            "score_admission_timeout_s": float(self.zo_admission_timeout_s),
            "max_score_admission_foreground_load": int(
                self.zo_admission_max_foreground_load
            ),
            "max_score_admission_gpu_utilization": float(
                self.zo_admission_max_gpu_utilization
            ),
            "score_admission_gpu_device": self.zo_admission_gpu_device or None,
            "zo_queue_token_rate": float(self.zo_queue_token_rate),
            "zo_queue_burst_tokens": int(self.zo_queue_burst_tokens),
            "zo_queue_max_inflight": int(self.zo_queue_max_inflight),
            "zo_queue_max_admitted_tokens": int(self.zo_queue_max_admitted_tokens),
            "zo_queue_poll_s": float(self.zo_queue_poll_s),
            "inter_step_delay_s": float(self.inter_step_delay_s),
            "initial_eval": self.initial_eval != "0",
            "enable_wandb": self.enable_wandb != "0",
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
        }

    def _resolve_reserved_lora_bank_bytes(self) -> int:
        manual = _env("VLLM_ZO_RESERVED_LORA_BANK_BYTES")
        if manual:
            parsed = int(manual)
            if parsed < 0:
                raise ValueError(
                    "VLLM_ZO_RESERVED_LORA_BANK_BYTES must be non-negative"
                )
            return parsed
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(
                self.model,
                trust_remote_code=self.trust_remote_code == "1",
            )
            direction_dtype = resolve_direction_dtype(self.direction_dtype, hf_config)
            metadata = build_lora_param_metadata_from_config(
                hf_config,
                device=torch.device("cpu"),
                dtype=direction_dtype,
                target_modules=self.lora_target_modules,
            )
            return estimate_lora_bank_state_memory_bytes(
                metadata,
                update_bank_rank=int(self.update_bank_rank),
                include_probe_u=True,
                include_pending_u=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to estimate serving ZO LoRA bank memory; set "
                "VLLM_ZO_RESERVED_LORA_BANK_BYTES manually to continue"
            ) from exc


def wait_server(config: ServingLoadConfig) -> None:
    timeout_s = int(_env("SERVER_WAIT_TIMEOUT", "600"))
    deadline = time.time() + timeout_s
    url = f"{config.base_url}/v1/models"
    while True:
        try:
            _json_request("GET", url, timeout=10.0)
            return
        except (HTTPError, URLError, TimeoutError):
            if time.time() > deadline:
                raise SystemExit(f"[serving-runner] server did not become ready: {url}")
            time.sleep(5)


def run_server(config: ServingLoadConfig) -> int:
    config.require_gpu()
    config.write_params()
    cmd = config.server_cmd()
    command_text = " ".join(shlex.quote(part) for part in cmd)
    print(f"[serving-runner] server command: {command_text}", flush=True)
    (config.log_root / "server.command.txt").write_text(
        f"[serving-runner] server command: {command_text}\n",
        encoding="utf-8",
    )
    return _stream_command(
        cmd,
        env=config.server_env(),
        log_path=config.log_root / "server.log",
    )


def run_bench(config: ServingLoadConfig, label: str) -> int:
    config.write_params()
    wait_server(config)
    return _stream_command(
        config.bench_cmd(label),
        env=_runtime_env(),
        log_path=config.log_root / f"{label}_rps{config.request_rate}.log",
    )


def start_zo(config: ServingLoadConfig) -> int:
    config.write_params()
    wait_server(config)
    response = _json_request(
        "POST",
        f"{config.base_url}/zo_vllm/serving_zo/start",
        payload=config.zo_payload(),
    )
    print(response)
    (config.log_root / "start_zo.response.json").write_text(
        response,
        encoding="utf-8",
    )
    return 0


def stop_zo(config: ServingLoadConfig) -> int:
    response = _json_request(
        "POST",
        f"{config.base_url}/zo_vllm/serving_zo/stop",
        payload={"wait": True, "timeout_s": 60},
    )
    print(response)
    config.log_root.mkdir(parents=True, exist_ok=True)
    (config.log_root / "stop_zo.response.json").write_text(response, encoding="utf-8")
    return 0


def status(
    config: ServingLoadConfig,
    *,
    output_name: str = "status.response.json",
) -> int:
    response = _json_request("GET", f"{config.base_url}/zo_vllm/serving_zo/status")
    print(response)
    config.log_root.mkdir(parents=True, exist_ok=True)
    (config.log_root / output_name).write_text(response, encoding="utf-8")
    return 0


def analyze(config: ServingLoadConfig) -> int:
    config.log_root.mkdir(parents=True, exist_ok=True)
    bench_json = Path(
        _env(
            "BENCH_JSON",
            str(config.log_root / f"baseline_rps{config.request_rate}.json"),
        )
    )
    zo_jsonl = _env("ZO_JSONL")
    if zo_jsonl:
        zo_jsonl_path = Path(zo_jsonl)
    else:
        model_slug = config.model.replace("/", "__")
        zo_jsonl_path = (
            PROJECT_ROOT
            / "phase7"
            / "logs"
            / "serving"
            / f"{model_slug}_{config.run_tag}"
            / "zo_metrics.jsonl"
        )
    summary_json = Path(
        _env("SUMMARY_JSON", str(config.log_root / "timeline_summary.json"))
    )
    status_code = _run_logged_python_script(
        [
            "phase7/scripts/analyze_serving_timeline.py",
            "--bench-json",
            str(bench_json),
            "--zo-jsonl",
            str(zo_jsonl_path),
            "--summary-json",
            str(summary_json),
        ],
        log_path=config.log_root / "analyze.log",
    )
    trace_path = Path(
        _env("SCHED_TRACE_JSONL", str(config.log_root / "scheduler_trace.jsonl"))
    )
    if trace_path.exists():
        status_code = status_code or _run_logged_python_script(
            [
                "phase7/scripts/analyze_scheduler_trace.py",
                "--trace",
                str(trace_path),
                "--out-json",
                str(config.log_root / "scheduler_trace_summary.json"),
                "--out-md",
                str(config.log_root / "scheduler_trace_summary.md"),
                "--max-target-ms",
                _env("SCHED_TRACE_MAX_TARGET_MS", "1000.0"),
            ],
            log_path=config.log_root / "scheduler_trace_analyze.log",
        )
    return status_code


def acceptance(config: ServingLoadConfig) -> int:
    baseline_json = _env("BASELINE_JSON")
    if not baseline_json:
        raise SystemExit(
            "[serving-runner] BASELINE_JSON is required for acceptance mode."
        )
    zo_summary_json = _env(
        "ZO_SUMMARY_JSON",
        str(config.log_root / "timeline_summary.json"),
    )
    return _run_logged_python_script(
        [
            "phase7/scripts/check_serving_acceptance.py",
            "--baseline-json",
            baseline_json,
            "--zo-summary-json",
            zo_summary_json,
            "--out-json",
            str(config.log_root / "acceptance.json"),
            "--min-throughput-ratio",
            _env("MIN_THROUGHPUT_RATIO", "0.98"),
            "--min-goodput-ratio",
            _env("MIN_GOODPUT_RATIO", "0.98"),
            "--max-p99-latency-delta-pct",
            _env("MAX_P99_LATENCY_DELTA_PCT", "5.0"),
            "--min-overlap-steps",
            _env("MIN_OVERLAP_STEPS", "1"),
        ],
        log_path=config.log_root / "acceptance.log",
    )


def summarize(config: ServingLoadConfig) -> int:
    config.log_root.mkdir(parents=True, exist_ok=True)
    summary_glob = _env(
        "SUMMARY_GLOB",
        str(
            PROJECT_ROOT / "phase7" / "logs" / "serving" / "*" / "timeline_summary.json"
        ),
    )
    summary_paths = sorted(glob.glob(summary_glob))
    if not summary_paths:
        raise SystemExit(
            f"[serving-runner] no timeline summaries matched: {summary_glob}"
        )
    args = [
        "phase7/scripts/summarize_serving_runs.py",
        *summary_paths,
        "--out-json",
        str(config.log_root / "serving_summary.json"),
        "--out-csv",
        str(config.log_root / "serving_summary.csv"),
    ]
    baseline_summary = _env("BASELINE_SUMMARY")
    if baseline_summary:
        args.extend(["--baseline-summary", baseline_summary])
    return _run_logged_python_script(args, log_path=config.log_root / "summarize.log")


def zo_bench(config: ServingLoadConfig) -> int:
    config.write_params()
    wait_server(config)
    start_zo(config)
    status_code = run_bench(config, "zo_on")
    if _env("STOP_ZO_AFTER_BENCH", "1") != "0":
        try:
            stop_status = stop_zo(config)
        except Exception as exc:
            print(f"[serving-runner] stop-zo failed: {exc}", flush=True)
            stop_status = 1
    else:
        try:
            stop_status = status(
                config,
                output_name="status_after_bench.json",
            )
        except Exception as exc:
            print(f"[serving-runner] status failed: {exc}", flush=True)
            stop_status = 1
    if status_code == 0:
        status_code = stop_status
    if status_code != 0:
        try:
            status(config, output_name="status_after_bench.json")
        except Exception:
            pass
        return status_code
    if _env("ANALYZE_AFTER_BENCH", "1") != "0":
        os.environ["BENCH_JSON"] = str(
            config.log_root / f"zo_on_rps{config.request_rate}.json"
        )
        status_code = status_code or analyze(config)
    if _env("BASELINE_JSON") and _env("ACCEPT_AFTER_BENCH", "1") != "0":
        os.environ["ZO_SUMMARY_JSON"] = str(config.log_root / "timeline_summary.json")
        status_code = status_code or acceptance(config)
    try:
        status(config, output_name="status_after_bench.json")
    except Exception:
        pass
    return status_code


def sweep(config: ServingLoadConfig) -> int:
    config.write_params()
    wait_server(config)
    status_code = 0
    for request_rate in _env("RPS_SWEEP", "0.5 1 2 4 6 8 10").split():
        os.environ["REQUEST_RATE"] = request_rate
        case_config = ServingLoadConfig(config.mode)
        status_code = status_code or run_bench(case_config, "sweep")
    return status_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="help",
        choices=[
            "help",
            "server",
            "bench",
            "sweep",
            "start-zo",
            "stop-zo",
            "status",
            "analyze",
            "summarize",
            "acceptance",
            "zo-bench",
        ],
    )
    args = parser.parse_args()
    if args.mode == "help":
        parser.print_help()
        return
    config = ServingLoadConfig(args.mode)
    handlers = {
        "server": run_server,
        "bench": lambda cfg: run_bench(cfg, "baseline"),
        "sweep": sweep,
        "start-zo": start_zo,
        "stop-zo": stop_zo,
        "status": status,
        "analyze": analyze,
        "summarize": summarize,
        "acceptance": acceptance,
        "zo-bench": zo_bench,
    }
    raise SystemExit(handlers[args.mode](config))


if __name__ == "__main__":
    main()
