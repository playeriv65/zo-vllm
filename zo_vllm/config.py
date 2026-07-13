"""Shared package-level defaults and runtime configuration for ZO-vLLM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ZO_PLUS_LORA_ID = 9001
DEFAULT_ZO_MINUS_LORA_ID = 9002
DEFAULT_ZO_MAX_LORAS = 2
DEFAULT_ZO_DTYPE = "float16"
DEFAULT_ZO_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_ZO_MODEL_NAME = "facebook/opt-2.7b"
DEFAULT_ZO_LEARNING_RATE = 1e-7
DEFAULT_ZO_WEIGHT_DECAY = 0.0
DEFAULT_ZO_RANK = 2
DEFAULT_ZO_NU = 50
DEFAULT_ZO_EPS = 1e-3
DEFAULT_ZO_DIRECTION_SCALE = 1.0
DEFAULT_ZO_PERTURBATION_NORMALIZATION = "rms"
DEFAULT_ZO_STEPS = 1000
DEFAULT_ZO_WARMUP_STEPS = 0
DEFAULT_ZO_BATCH_SIZE = 16
DEFAULT_ZO_NUM_SAMPLES = 1000
DEFAULT_ZO_NUM_DEV = 500
DEFAULT_ZO_EVAL_INTERVAL = 100
DEFAULT_ZO_EVAL_ACCURACY_SAMPLES = 512
DEFAULT_ZO_SEED = 42
DEFAULT_ZO_DATA_SEED = None
DEFAULT_ZO_TRAIN_SAMPLER = "sequential"
DEFAULT_ZO_TASK_SHUFFLE_IMPL = "numpy"
DEFAULT_ZO_TRAIN_OBJECTIVE = "sst2_classification"
DEFAULT_ZO_RANDOM_DEVICE = "cuda"
DEFAULT_ZO_DIRECTION_SAMPLING = "exact"
DEFAULT_ZO_DIRECTION_PROVIDER = "lozo"
DEFAULT_ZO_LOZO_PROVIDER_MODE = "fast"
DEFAULT_ZO_TRAIN_SCOPE = "lora_normal"
DEFAULT_ZO_PERTURB_EMBEDDINGS = "0"
DEFAULT_ZO_ENFORCE_EAGER = "0"
DEFAULT_ZO_WEIGHT_UPDATE = "direct"
DEFAULT_ZO_WEIGHT_UPDATE_PRECISION = "param"
DEFAULT_ZO_DIRECT_UPDATE_MODE = "accumulate"
DEFAULT_ZO_QUANTIZED_UPDATE_MODE = "none"
DEFAULT_ZO_UPDATE_BANK_RANK = "auto"
DEFAULT_ZO_GRADIENT_ACCUMULATION_UPDATE_STEPS = 0
DEFAULT_ZO_U_BETA = 1.0
DEFAULT_ZO_U_NORM_CAP = None
DEFAULT_ZO_QKV_WEIGHT_UPDATE = "batched"
DEFAULT_ZO_SYNC_WEIGHT_UPDATE = "0"
DEFAULT_ZO_SCORING_BACKEND = "direct_worker"
DEFAULT_ZO_DIRECT_WORKER_MAX_LOGITS_TOKENS = 8192
DEFAULT_ZO_DIRECT_WORKER_LOSS_IMPL = "logprobs"
DEFAULT_ZO_BASE_EVAL_MODE = "generate"
DEFAULT_ZO_PROFILE_MODE = "minimal"
DEFAULT_ZO_DIRECT_LORA_FROM_DIRECTIONS = "1"
DEFAULT_ZO_ACCURACY_EVAL_MODE = "auto"
DEFAULT_ZO_MAX_NEW_TOKENS = 50
DEFAULT_ZO_WANDB_PROJECT = "zo-vllm"
DEFAULT_ZO_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
)
DEFAULT_ZO_WANDB_ENTITY = "playeriv65-university-of-minnesota"


def _validate_zo_nu(nu: int) -> int:
    nu_i = int(nu)
    if nu_i == -1:
        return nu_i
    if nu_i <= 0:
        raise ValueError("nu must be positive or -1 for infinite reuse")
    return nu_i


def _normalize_zo_perturbation_normalization(value: str | bool | None) -> str:
    if value is None:
        return DEFAULT_ZO_PERTURBATION_NORMALIZATION
    if isinstance(value, bool):
        return DEFAULT_ZO_PERTURBATION_NORMALIZATION if value else "none"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"1", "true", "yes", "on", "rms", "rms_per_weight"}:
        return "rms"
    if normalized in {"0", "false", "no", "off", "none", "disabled"}:
        return "none"
    raise ValueError(f"unknown perturbation_normalization: {value}")


@dataclass(frozen=True)
class ZOVLLMSlotConfig:
    """Persistent LoRA slot ids used for plus/minus ZO perturbations."""

    plus_id: int = DEFAULT_ZO_PLUS_LORA_ID
    minus_id: int = DEFAULT_ZO_MINUS_LORA_ID
    max_loras: int = DEFAULT_ZO_MAX_LORAS

    def __post_init__(self) -> None:
        plus_id = int(self.plus_id)
        minus_id = int(self.minus_id)
        max_loras = int(self.max_loras)
        if plus_id <= 0:
            raise ValueError("plus_id must be positive")
        if minus_id <= 0:
            raise ValueError("minus_id must be positive")
        if plus_id == minus_id:
            raise ValueError("plus_id and minus_id must be different")
        if max_loras < 2:
            raise ValueError("max_loras must be at least 2")
        object.__setattr__(self, "plus_id", plus_id)
        object.__setattr__(self, "minus_id", minus_id)
        object.__setattr__(self, "max_loras", max_loras)


@dataclass(frozen=True)
class ZOVLLMEngineConfig:
    """Runtime defaults for constructing the reusable vLLM-backed engine."""

    max_model_len: int | None = None
    gpu_memory_utilization: float = DEFAULT_ZO_GPU_MEMORY_UTILIZATION
    enforce_eager: bool = False
    dtype: str = DEFAULT_ZO_DTYPE
    lora_rank: int | None = None
    target_modules: Sequence[str] | str = field(
        default_factory=lambda: DEFAULT_ZO_TARGET_MODULES
    )
    slot: ZOVLLMSlotConfig = field(default_factory=ZOVLLMSlotConfig)
    zo_reserved_gpu_bytes: int = 0
    llm_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        max_model_len = None if self.max_model_len is None else int(self.max_model_len)
        gpu_memory_utilization = float(self.gpu_memory_utilization)
        lora_rank = None if self.lora_rank is None else int(self.lora_rank)
        zo_reserved_gpu_bytes = int(self.zo_reserved_gpu_bytes)
        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be positive when provided")
        if gpu_memory_utilization <= 0.0:
            raise ValueError("gpu_memory_utilization must be positive")
        if lora_rank is not None and lora_rank <= 0:
            raise ValueError("lora_rank must be positive when provided")
        if zo_reserved_gpu_bytes < 0:
            raise ValueError("zo_reserved_gpu_bytes must be non-negative")
        if isinstance(self.target_modules, str):
            modules = tuple(_parse_target_modules(self.target_modules) or ())
        else:
            modules = tuple(str(item).strip() for item in self.target_modules)
            modules = tuple(item for item in modules if item)
        if not modules:
            raise ValueError("target_modules must not be empty")
        object.__setattr__(self, "max_model_len", max_model_len)
        object.__setattr__(self, "gpu_memory_utilization", gpu_memory_utilization)
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "lora_rank", lora_rank)
        object.__setattr__(self, "target_modules", modules)
        object.__setattr__(self, "zo_reserved_gpu_bytes", zo_reserved_gpu_bytes)
        if not isinstance(self.slot, ZOVLLMSlotConfig):
            raise TypeError("slot must be a ZOVLLMSlotConfig")
        if self.llm_kwargs is not None:
            object.__setattr__(self, "llm_kwargs", dict(self.llm_kwargs))


@dataclass(frozen=True)
class VLLMZOConfig:
    """Algorithm settings for one vLLM-backed ZO training loop."""

    estimator: str = "single_direction_antithetic"
    num_queries: int = 1
    perturbation_sides: str = "two_sided"
    query_microbatch_size: int = 2
    multi_query_direction_mode: str = "shared_basis"
    population_size: int = 30
    sigma: float = DEFAULT_ZO_EPS
    reward_shaping: str = "z_score"
    direction_provider: str = DEFAULT_ZO_DIRECTION_PROVIDER
    lozo_provider_mode: str = DEFAULT_ZO_LOZO_PROVIDER_MODE
    rank: int = DEFAULT_ZO_RANK
    nu: int = DEFAULT_ZO_NU
    eps: float = DEFAULT_ZO_EPS
    random_device: str = DEFAULT_ZO_RANDOM_DEVICE
    direction_sampling: str = DEFAULT_ZO_DIRECTION_SAMPLING
    v_normalization: str = "none"
    power_iter_steps: int = 5
    low_rank_oversample: int = 4
    basis_method: str = "power_iter"
    direction_scale: float | None = DEFAULT_ZO_DIRECTION_SCALE
    perturbation_normalization: str = DEFAULT_ZO_PERTURBATION_NORMALIZATION
    u_dim: int | None = None
    u_pool_seed_offset: int = 300000
    max_logits_tokens: int = DEFAULT_ZO_DIRECT_WORKER_MAX_LOGITS_TOKENS
    loss_impl: str = DEFAULT_ZO_DIRECT_WORKER_LOSS_IMPL
    score_chunk_size: int = 0
    activation_force_eager: bool = True
    basis_seed_offset: int = 100000
    perturb_seed_offset: int = 200000
    seed: int = DEFAULT_ZO_SEED

    def __post_init__(self) -> None:
        estimator = str(self.estimator)
        direction_provider = str(self.direction_provider)
        lozo_provider_mode = str(self.lozo_provider_mode)
        if estimator not in {
            "single_direction_antithetic",
            "multi_query",
            "evolution_strategy",
        }:
            raise ValueError(f"unsupported estimator: {estimator}")
        if int(self.num_queries) <= 0:
            raise ValueError("num_queries must be positive")
        perturbation_sides = str(self.perturbation_sides).replace("-", "_")
        if perturbation_sides not in {"two_sided", "one_sided"}:
            raise ValueError("perturbation_sides must be two_sided or one_sided")
        if int(self.query_microbatch_size) <= 0:
            raise ValueError("query_microbatch_size must be positive")
        multi_query_direction_mode = str(self.multi_query_direction_mode).replace(
            "-", "_"
        )
        if multi_query_direction_mode not in {"shared_basis", "independent"}:
            raise ValueError(
                "multi_query_direction_mode must be shared_basis or independent"
            )
        if int(self.population_size) <= 0:
            raise ValueError("population_size must be positive")
        if float(self.sigma) <= 0.0:
            raise ValueError("sigma must be positive")
        reward_shaping = str(self.reward_shaping).replace("-", "_")
        if reward_shaping == "z_scores":
            reward_shaping = "z_score"
        if reward_shaping not in {"z_score", "none"}:
            raise ValueError("reward_shaping must be z_score or none")
        if direction_provider not in {"lozo", "agzo", "uagzo", "suagzo"}:
            raise ValueError(f"unsupported direction_provider: {direction_provider}")
        if lozo_provider_mode not in {"fast", "scheduled"}:
            raise ValueError(f"unsupported lozo_provider_mode: {lozo_provider_mode}")
        if int(self.rank) <= 0:
            raise ValueError("rank must be positive")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if self.random_device not in {"cpu", "cuda"}:
            raise ValueError(f"unknown random_device: {self.random_device}")
        if self.direction_sampling not in {"exact", "flat"}:
            raise ValueError(f"unknown direction_sampling: {self.direction_sampling}")
        if self.v_normalization not in {"none", "unit"}:
            raise ValueError(f"unknown v_normalization: {self.v_normalization}")
        if int(self.power_iter_steps) <= 0:
            raise ValueError("power_iter_steps must be positive")
        if int(self.low_rank_oversample) < 0:
            raise ValueError("low_rank_oversample must be non-negative")
        if self.basis_method not in {"power_iter", "svd", "low_rank_svd"}:
            raise ValueError(f"unsupported basis_method: {self.basis_method}")
        direction_scale = 1.0 if self.direction_scale is None else self.direction_scale
        if float(direction_scale) < 0.0:
            raise ValueError("direction_scale must be non-negative")
        if int(self.score_chunk_size) < 0:
            raise ValueError("score_chunk_size must be non-negative")
        if direction_provider in {"uagzo", "suagzo"}:
            if self.u_dim is None or int(self.u_dim) <= 0:
                raise ValueError(f"u_dim must be positive for {direction_provider}")
            if direction_provider == "uagzo" and int(self.u_dim) < int(self.rank):
                raise ValueError("u_dim must be greater than or equal to rank")
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "num_queries", int(self.num_queries))
        object.__setattr__(self, "perturbation_sides", perturbation_sides)
        object.__setattr__(
            self, "query_microbatch_size", int(self.query_microbatch_size)
        )
        object.__setattr__(
            self, "multi_query_direction_mode", multi_query_direction_mode
        )
        object.__setattr__(self, "population_size", int(self.population_size))
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "reward_shaping", reward_shaping)
        object.__setattr__(self, "direction_provider", direction_provider)
        object.__setattr__(self, "lozo_provider_mode", lozo_provider_mode)
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "nu", _validate_zo_nu(self.nu))
        object.__setattr__(self, "eps", float(self.eps))
        object.__setattr__(self, "random_device", str(self.random_device))
        object.__setattr__(self, "direction_sampling", str(self.direction_sampling))
        object.__setattr__(self, "v_normalization", str(self.v_normalization))
        object.__setattr__(self, "direction_scale", float(direction_scale))
        object.__setattr__(
            self,
            "perturbation_normalization",
            _normalize_zo_perturbation_normalization(self.perturbation_normalization),
        )
        object.__setattr__(self, "score_chunk_size", int(self.score_chunk_size))
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class ZOServingConfig:
    """Serving-time ZO trainer defaults shared by API and experiment runners."""

    task_name: str = "sst2"
    steps: int = 100
    batch_size: int = 8
    num_train: int = 1024
    num_dev: int = 256
    eval_interval: int = 50
    rank: int = DEFAULT_ZO_RANK
    update_bank_rank: int | str | None = "auto"
    eps: float = 1e-3
    learning_rate: float = DEFAULT_ZO_LEARNING_RATE
    weight_decay: float = 0.0
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    nu: int = 50
    seed: int = 42
    priority: int = 1000
    gradient_accumulation_update_steps: int = 0
    u_beta: float = 1.0
    u_norm_cap: float | None = None
    random_device: str = "cuda"
    direction_device: str = "cuda"
    direction_dtype: str = "auto"
    direction_sampling: str = "exact"
    direction_scale: float | None = DEFAULT_ZO_DIRECTION_SCALE
    perturbation_normalization: str = DEFAULT_ZO_PERTURBATION_NORMALIZATION
    v_normalization: str = "none"
    target_modules: Sequence[str] | str | None = None
    include_lm_head: bool = False
    include_embeddings: bool = False
    inter_step_delay_s: float = 0.0
    slot_write_stream: str = "background_sync"
    score_admission_policy: str = "idle_gap"
    score_admission_poll_s: float = 0.005
    score_admission_timeout_s: float = 0.0
    max_score_admission_foreground_load: int = 0
    max_score_admission_gpu_utilization: float = 70.0
    score_admission_gpu_device: str | None = None
    zo_queue_token_rate: float = 1024.0
    zo_queue_burst_tokens: int = 512
    zo_queue_max_inflight: int = 64
    zo_queue_max_admitted_tokens: int = 1024
    zo_queue_poll_s: float = 0.01
    initial_eval: bool = True
    enable_wandb: bool = True
    wandb_project: str = "zo-vllm-serving-zo"
    wandb_entity: str = DEFAULT_ZO_WANDB_ENTITY
    output_dir: str = "zo_vllm_runs/serving"
    stop_timeout_s: float = 30.0
    slot: ZOVLLMSlotConfig = field(default_factory=ZOVLLMSlotConfig)

    def __post_init__(self) -> None:
        positive_int_fields = {
            "batch_size": self.batch_size,
            "num_train": self.num_train,
            "num_dev": self.num_dev,
            "rank": self.rank,
            "nu": self.nu,
            "priority": self.priority,
            "zo_queue_burst_tokens": self.zo_queue_burst_tokens,
            "zo_queue_max_inflight": self.zo_queue_max_inflight,
            "zo_queue_max_admitted_tokens": self.zo_queue_max_admitted_tokens,
        }
        for name, value in positive_int_fields.items():
            parsed = int(value)
            if parsed <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, parsed)
        non_negative_int_fields = {
            "steps": self.steps,
            "eval_interval": self.eval_interval,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_update_steps": (
                self.gradient_accumulation_update_steps
            ),
            "max_score_admission_foreground_load": (
                self.max_score_admission_foreground_load
            ),
        }
        for name, value in non_negative_int_fields.items():
            parsed = int(value)
            if parsed < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, parsed)
        non_negative_float_fields = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "inter_step_delay_s": self.inter_step_delay_s,
            "score_admission_poll_s": self.score_admission_poll_s,
            "score_admission_timeout_s": self.score_admission_timeout_s,
            "zo_queue_poll_s": self.zo_queue_poll_s,
            "stop_timeout_s": self.stop_timeout_s,
        }
        for name, value in non_negative_float_fields.items():
            parsed = float(value)
            if parsed < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, parsed)
        positive_float_fields = {
            "eps": self.eps,
            "u_beta": self.u_beta,
            "max_score_admission_gpu_utilization": (
                self.max_score_admission_gpu_utilization
            ),
            "zo_queue_token_rate": self.zo_queue_token_rate,
        }
        for name, value in positive_float_fields.items():
            parsed = float(value)
            if parsed <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, parsed)
        if self.u_beta > 1.0:
            raise ValueError("u_beta must be <= 1.0")
        if self.u_norm_cap is not None and float(self.u_norm_cap) <= 0.0:
            raise ValueError("u_norm_cap must be positive when provided")
        if float(self.max_score_admission_gpu_utilization) > 100.0:
            raise ValueError("max_score_admission_gpu_utilization must be <= 100.0")
        if self.direction_scale is not None:
            object.__setattr__(self, "direction_scale", float(self.direction_scale))
        object.__setattr__(self, "output_dir", str(self.output_dir))
        if str(self.task_name).strip().lower() != "sst2":
            raise ValueError("serving task_name currently supports only sst2")
        object.__setattr__(self, "task_name", "sst2")
        object.__setattr__(self, "lr_scheduler_type", str(self.lr_scheduler_type))
        object.__setattr__(
            self,
            "perturbation_normalization",
            str(self.perturbation_normalization),
        )
        if not isinstance(self.slot, ZOVLLMSlotConfig):
            raise TypeError("slot must be a ZOVLLMSlotConfig")
        if self.target_modules is not None:
            modules = _parse_target_modules(self.target_modules)
            object.__setattr__(self, "target_modules", tuple(modules or ()))


DEFAULT_ZO_SERVING_CONFIG = ZOServingConfig()


def _parse_target_modules(
    value: str | list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    modules: list[str] = []
    for chunk in str(value).replace(",", " ").split():
        item = chunk.strip()
        if item:
            modules.append(item)
    return modules or None


__all__ = [
    "DEFAULT_ZO_DTYPE",
    "DEFAULT_ZO_ACCURACY_EVAL_MODE",
    "DEFAULT_ZO_BASE_EVAL_MODE",
    "DEFAULT_ZO_BATCH_SIZE",
    "DEFAULT_ZO_DATA_SEED",
    "DEFAULT_ZO_DIRECT_LORA_FROM_DIRECTIONS",
    "DEFAULT_ZO_DIRECT_UPDATE_MODE",
    "DEFAULT_ZO_DIRECT_WORKER_LOSS_IMPL",
    "DEFAULT_ZO_DIRECT_WORKER_MAX_LOGITS_TOKENS",
    "DEFAULT_ZO_DIRECTION_PROVIDER",
    "DEFAULT_ZO_DIRECTION_SAMPLING",
    "DEFAULT_ZO_DIRECTION_SCALE",
    "DEFAULT_ZO_ENFORCE_EAGER",
    "DEFAULT_ZO_EPS",
    "DEFAULT_ZO_EVAL_ACCURACY_SAMPLES",
    "DEFAULT_ZO_EVAL_INTERVAL",
    "DEFAULT_ZO_GRADIENT_ACCUMULATION_UPDATE_STEPS",
    "DEFAULT_ZO_GPU_MEMORY_UTILIZATION",
    "DEFAULT_ZO_LEARNING_RATE",
    "DEFAULT_ZO_LOZO_PROVIDER_MODE",
    "DEFAULT_ZO_MAX_NEW_TOKENS",
    "DEFAULT_ZO_MAX_LORAS",
    "DEFAULT_ZO_MODEL_NAME",
    "DEFAULT_ZO_MINUS_LORA_ID",
    "DEFAULT_ZO_NUM_DEV",
    "DEFAULT_ZO_NUM_SAMPLES",
    "DEFAULT_ZO_PERTURBATION_NORMALIZATION",
    "DEFAULT_ZO_PERTURB_EMBEDDINGS",
    "DEFAULT_ZO_PLUS_LORA_ID",
    "DEFAULT_ZO_PROFILE_MODE",
    "DEFAULT_ZO_QKV_WEIGHT_UPDATE",
    "DEFAULT_ZO_QUANTIZED_UPDATE_MODE",
    "DEFAULT_ZO_RANDOM_DEVICE",
    "DEFAULT_ZO_RANK",
    "DEFAULT_ZO_SCORING_BACKEND",
    "DEFAULT_ZO_SEED",
    "DEFAULT_ZO_SERVING_CONFIG",
    "DEFAULT_ZO_STEPS",
    "DEFAULT_ZO_SYNC_WEIGHT_UPDATE",
    "DEFAULT_ZO_TARGET_MODULES",
    "DEFAULT_ZO_TRAIN_OBJECTIVE",
    "DEFAULT_ZO_TRAIN_SAMPLER",
    "DEFAULT_ZO_TRAIN_SCOPE",
    "DEFAULT_ZO_U_BETA",
    "DEFAULT_ZO_U_NORM_CAP",
    "DEFAULT_ZO_UPDATE_BANK_RANK",
    "DEFAULT_ZO_WANDB_PROJECT",
    "DEFAULT_ZO_WANDB_ENTITY",
    "DEFAULT_ZO_WARMUP_STEPS",
    "DEFAULT_ZO_WEIGHT_DECAY",
    "DEFAULT_ZO_WEIGHT_UPDATE",
    "DEFAULT_ZO_WEIGHT_UPDATE_PRECISION",
    "DEFAULT_ZO_NU",
    "ZOServingConfig",
    "ZOVLLMEngineConfig",
    "ZOVLLMSlotConfig",
]
