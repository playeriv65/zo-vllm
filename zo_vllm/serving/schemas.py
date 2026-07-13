from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from zo_vllm.config import DEFAULT_ZO_SERVING_CONFIG

DEFAULT_REQUEST = DEFAULT_ZO_SERVING_CONFIG
SUPPORTED_SCORE_ADMISSION_POLICIES = {
    "gpu_utilization",
    "idle_gap",
    "queued",
    "scheduler_only",
}


class ServingZOStartRequest(BaseModel):
    """Request body for starting HF-native ZO training during serving."""

    task_name: Literal["sst2"] = DEFAULT_REQUEST.task_name
    run_name: str | None = None
    output_dir: str = DEFAULT_REQUEST.output_dir
    steps: int = Field(default=DEFAULT_REQUEST.steps, ge=0)
    batch_size: int = Field(default=DEFAULT_REQUEST.batch_size, ge=1)
    num_train: int = Field(default=DEFAULT_REQUEST.num_train, ge=1)
    num_dev: int = Field(default=DEFAULT_REQUEST.num_dev, ge=1)
    eval_interval: int = Field(default=DEFAULT_REQUEST.eval_interval, ge=0)
    rank: int = Field(default=DEFAULT_REQUEST.rank, ge=1)
    update_bank_rank: int | str | None = DEFAULT_REQUEST.update_bank_rank
    eps: float = Field(default=DEFAULT_REQUEST.eps, gt=0.0)
    learning_rate: float = Field(default=DEFAULT_REQUEST.learning_rate, ge=0.0)
    weight_decay: float = Field(default=DEFAULT_REQUEST.weight_decay, ge=0.0)
    lr_scheduler_type: str = DEFAULT_REQUEST.lr_scheduler_type
    warmup_steps: int = Field(default=DEFAULT_REQUEST.warmup_steps, ge=0)
    nu: int = Field(default=DEFAULT_REQUEST.nu, ge=1)
    seed: int = DEFAULT_REQUEST.seed
    data_seed: int | None = None
    priority: int = DEFAULT_REQUEST.priority
    plus_id: int = Field(default=DEFAULT_REQUEST.slot.plus_id, ge=1)
    minus_id: int = Field(default=DEFAULT_REQUEST.slot.minus_id, ge=1)
    gradient_accumulation_update_steps: int = Field(
        default=DEFAULT_REQUEST.gradient_accumulation_update_steps,
        ge=0,
    )
    u_beta: float = Field(default=DEFAULT_REQUEST.u_beta, ge=0.0, le=1.0)
    u_norm_cap: float | None = Field(default=DEFAULT_REQUEST.u_norm_cap, gt=0.0)
    random_device: Literal["cpu", "cuda"] = DEFAULT_REQUEST.random_device
    direction_device: Literal["cpu", "cuda"] = DEFAULT_REQUEST.direction_device
    direction_dtype: str = DEFAULT_REQUEST.direction_dtype
    direction_sampling: Literal["exact", "flat"] = DEFAULT_REQUEST.direction_sampling
    direction_scale: float | None = Field(
        default=DEFAULT_REQUEST.direction_scale,
        ge=0.0,
    )
    perturbation_normalization: Literal["rms", "none"] = (
        DEFAULT_REQUEST.perturbation_normalization
    )
    v_normalization: Literal["none", "unit"] = DEFAULT_REQUEST.v_normalization
    target_modules: list[str] | str | None = DEFAULT_REQUEST.target_modules
    include_lm_head: bool = DEFAULT_REQUEST.include_lm_head
    include_embeddings: bool = DEFAULT_REQUEST.include_embeddings
    inter_step_delay_s: float = Field(
        default=DEFAULT_REQUEST.inter_step_delay_s,
        ge=0.0,
    )
    slot_write_stream: str = DEFAULT_REQUEST.slot_write_stream
    score_admission_policy: str = DEFAULT_REQUEST.score_admission_policy
    score_admission_poll_s: float = Field(
        default=DEFAULT_REQUEST.score_admission_poll_s,
        ge=0.0,
    )
    score_admission_timeout_s: float = Field(
        default=DEFAULT_REQUEST.score_admission_timeout_s,
        ge=0.0,
    )
    max_score_admission_foreground_load: int = Field(
        default=DEFAULT_REQUEST.max_score_admission_foreground_load,
        ge=0,
    )
    max_score_admission_gpu_utilization: float = Field(
        default=DEFAULT_REQUEST.max_score_admission_gpu_utilization,
        ge=0.0,
        le=100.0,
    )
    score_admission_gpu_device: str | None = DEFAULT_REQUEST.score_admission_gpu_device
    zo_queue_token_rate: float = Field(
        default=DEFAULT_REQUEST.zo_queue_token_rate,
        gt=0.0,
    )
    zo_queue_burst_tokens: int = Field(
        default=DEFAULT_REQUEST.zo_queue_burst_tokens,
        ge=1,
    )
    zo_queue_max_inflight: int = Field(
        default=DEFAULT_REQUEST.zo_queue_max_inflight,
        ge=1,
    )
    zo_queue_max_admitted_tokens: int = Field(
        default=DEFAULT_REQUEST.zo_queue_max_admitted_tokens,
        ge=1,
    )
    zo_queue_poll_s: float = Field(default=DEFAULT_REQUEST.zo_queue_poll_s, ge=0.0)
    initial_eval: bool = DEFAULT_REQUEST.initial_eval
    enable_wandb: bool = DEFAULT_REQUEST.enable_wandb
    wandb_project: str = DEFAULT_REQUEST.wandb_project
    wandb_entity: str = DEFAULT_REQUEST.wandb_entity


class ServingZOStopRequest(BaseModel):
    wait: bool = True
    timeout_s: float = Field(default=DEFAULT_REQUEST.stop_timeout_s, ge=0.0)
