"""Phase 3 HF Trainer speed migration probe.

This script is intentionally phase-local. It uses the generic ``zo_trainer``
surface exactly like a Hugging Face Trainer script, then computes Phase 3 timing
summaries from a script-local callback.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoConfig, TrainerCallback

from zo_trainer import (
    VLLMDataCollator,
    ZOTrainer,
    ZOTrainerArguments,
    build_target_lm_preprocess,
)
from zo_vllm.tasks.hf_preprocessing import build_sst2_prompt_classification_preprocess
from zo_vllm.tasks import TaskConfig, get_task
from zo_vllm.tasks.hf_preprocessing import build_objective_hf_dataset
from zo_vllm.config import VLLMZOConfig, ZOVLLMEngineConfig
from zo_vllm.core.lora_scope import resolve_lora_target_modules
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.training.update_state import AccumulatedLowRankUpdateState
from zo_vllm.training import VLLMZOModel, build_tokenizer


DEFAULT_MODEL = "facebook/opt-13b"
DEFAULT_PHASE3_DATA_SEED = 42
DEFAULT_PHASE3_STEPS = 300
DEFAULT_PHASE3_TAIL_STEPS = 100
DEFAULT_PHASE3_BATCH_SIZE = 16
DEFAULT_PHASE3_DATASET_REPEATS = 1
DEFAULT_PHASE3_NUM_TRAIN = 1000
DEFAULT_PHASE3_NUM_DEV = 500
DEFAULT_PHASE3_RANK = 2
DEFAULT_PHASE3_NU = 50
DEFAULT_PHASE3_LR = 3e-7
DEFAULT_PHASE3_MAX_LENGTH = 2048
DEFAULT_PHASE3_MAX_NUM_BATCHED_TOKENS = 16384
DEFAULT_PHASE3_MAX_LOGITS_TOKENS = 8192
TARGET_LM_TASKS = {"synthetic", "sst2"}
DEFAULT_PROMPTS = (
    ("Review: good movie\nSentiment:", " positive"),
    ("Review: bad movie\nSentiment:", " negative"),
    ("Review: excellent acting\nSentiment:", " positive"),
    ("Review: boring story\nSentiment:", " negative"),
)


class Phase3StepTimingCallback(TrainerCallback):
    """Collect per-step wall time for Phase 3 summaries."""

    def __init__(self, *, warmup_steps: int = 0) -> None:
        self.warmup_steps = int(warmup_steps)
        self._step_start: float | None = None
        self.step_s: list[float] = []

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, state, control, kwargs
        self._step_start = time.perf_counter()
        return None

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, control, kwargs
        if self._step_start is not None:
            raw_step = int(state.global_step)
            if raw_step > self.warmup_steps:
                self.step_s.append(float(time.perf_counter() - self._step_start))
        self._step_start = None
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default="sst2")
    parser.add_argument(
        "--task-objective",
        choices=["registered", "target_lm", "sst2_classification"],
        default="registered",
        help="HF-native token-batch objective used by this speed probe.",
    )
    parser.add_argument("--output-root", default="phase3/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_PHASE3_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--tail-steps", type=int, default=DEFAULT_PHASE3_TAIL_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PHASE3_BATCH_SIZE)
    parser.add_argument(
        "--dataset-repeats", type=int, default=DEFAULT_PHASE3_DATASET_REPEATS
    )
    parser.add_argument("--num-train", type=int, default=DEFAULT_PHASE3_NUM_TRAIN)
    parser.add_argument("--num-dev", type=int, default=DEFAULT_PHASE3_NUM_DEV)
    parser.add_argument("--max-length", type=int, default=DEFAULT_PHASE3_MAX_LENGTH)
    parser.add_argument("--rank", type=int, default=DEFAULT_PHASE3_RANK)
    parser.add_argument("--nu", type=int, default=DEFAULT_PHASE3_NU)
    parser.add_argument("--lr", type=float, default=DEFAULT_PHASE3_LR)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--estimator", default="single_direction_antithetic")
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument(
        "--perturbation-sides",
        choices=["two_sided", "two-sided", "one_sided", "one-sided"],
        default="two_sided",
    )
    parser.add_argument("--query-microbatch-size", type=int, default=2)
    parser.add_argument("--multi-query-direction-mode", default="shared_basis")
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--reward-shaping", default="z_score")
    parser.add_argument(
        "--direction-provider",
        choices=["lozo", "agzo", "uagzo", "suagzo"],
        default="lozo",
    )
    parser.add_argument(
        "--lozo-provider-mode",
        choices=["fast", "scheduled"],
        default="fast",
    )
    parser.add_argument("--random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--direction-sampling",
        choices=["exact", "flat"],
        default="flat",
    )
    parser.add_argument("--direction-scale", type=float, default=1.0)
    parser.add_argument(
        "--perturbation-normalization",
        choices=["rms", "none"],
        default="rms",
    )
    parser.add_argument("--v-normalization", choices=["none", "unit"], default="none")
    parser.add_argument("--u-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_PHASE3_DATA_SEED)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_PHASE3_DATA_SEED)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_PHASE3_MAX_LENGTH)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_PHASE3_MAX_NUM_BATCHED_TOKENS,
        help="vLLM max_num_batched_tokens. Omit to use vLLM native auto selection.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="vLLM max_num_seqs. Use 0 to leave the vLLM default unset.",
    )
    parser.add_argument(
        "--max-logits-tokens",
        type=int,
        default=DEFAULT_PHASE3_MAX_LOGITS_TOKENS,
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument(
        "--direct-update-mode",
        choices=["direct", "accumulate"],
        default="accumulate",
    )
    parser.add_argument("--weight-update-precision", default="param")
    parser.add_argument("--qkv-weight-update", default="batched")
    parser.add_argument("--gradient-accumulation-update-steps", type=int, default=0)
    parser.add_argument("--u-beta", type=float, default=1.0)
    parser.add_argument("--u-norm-cap", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")
    if args.tail_steps <= 0:
        raise SystemExit("--tail-steps must be positive")
    if args.dataset_repeats <= 0:
        raise SystemExit("--dataset-repeats must be positive")
    if str(args.task_objective) == "registered":
        get_task(str(args.task))
    elif (
        str(args.task_objective) == "target_lm"
        and str(args.task) not in TARGET_LM_TASKS
    ):
        raise SystemExit(
            "--task-objective target_lm currently supports --task synthetic or sst2"
        )
    if str(args.task_objective) == "sst2_classification":
        if str(args.task) != "sst2":
            raise SystemExit(
                "--task-objective sst2_classification requires --task sst2"
            )
    return args


def main() -> None:
    args = parse_args()
    _prepare_environment()
    run_id = args.run_id or (
        "hf_trainer_speed_"
        f"{_safe_name(args.model)}_s{args.steps}_b{args.batch_size}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    engine: ZOVLLMEngine | None = None
    started = time.time()
    try:
        engine, zo_model, tokenizer = _build_zo_model(args)
        train_dataset, data_collator, trainer_cls = _build_training_inputs(
            tokenizer,
            args,
        )
        _set_runner_seeds(int(args.seed))
        timing = Phase3StepTimingCallback(warmup_steps=int(args.warmup_steps))
        trainer = trainer_cls(
            model=zo_model,
            args=ZOTrainerArguments(
                output_dir=str(output_dir / "hf_output"),
                max_steps=int(args.steps) + int(args.warmup_steps),
                per_device_train_batch_size=int(args.batch_size),
                logging_steps=max(1, int(args.logging_steps)),
                eval_strategy="no",
                save_strategy="no",
                disable_tqdm=True,
                report_to=[],
                remove_unused_columns=False,
                learning_rate=float(args.lr),
                lr_scheduler_type=str(args.lr_scheduler_type),
            ),
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=[timing],
        )
        trainer._phase3_warmup_steps = int(args.warmup_steps)
        print(
            "[phase3-hf] parameters=" + json.dumps(_config_dict(args), sort_keys=True),
            flush=True,
        )
        output = trainer.train()
        phase3_profile, step_records = _profile_from_log_history(
            trainer.state.log_history,
            step_s=timing.step_s,
        )
        result = _build_result(
            args=args,
            output_metrics=output.metrics,
            step_s=timing.step_s,
            phase3_profile=phase3_profile,
            step_records=step_records,
            log_history=trainer.state.log_history,
            wall_clock_s=time.time() - started,
        )
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("[phase3-hf] result=" + json.dumps(result, sort_keys=True), flush=True)
        print(f"[phase3-hf] wrote {result_path}", flush=True)
    finally:
        _cleanup_engine(engine)


def _prepare_environment() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("WANDB_MODE", "offline")


def _set_runner_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_zo_model(args: argparse.Namespace) -> tuple[ZOVLLMEngine, VLLMZOModel, Any]:
    model_config = AutoConfig.from_pretrained(args.model)
    tokenizer = build_tokenizer(args.model, use_fast=False)
    target_modules = resolve_lora_target_modules(None)
    engine = ZOVLLMEngine(
        model=args.model,
        rank=int(args.rank),
        model_config=model_config,
        config=ZOVLLMEngineConfig(
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            enforce_eager=bool(int(args.enforce_eager)),
            lora_rank=int(args.rank),
            target_modules=target_modules,
            llm_kwargs=_build_llm_kwargs(args),
        ),
    )
    weight_sync = WeightSync(
        engine.llm,
        num_layers=model_config.num_hidden_layers,
        model_config=model_config,
    )
    update_state = _build_update_state(args, engine=engine, weight_sync=weight_sync)
    zo_model = VLLMZOModel(
        engine=engine,
        weight_sync=weight_sync,
        config=VLLMZOConfig(
            estimator=str(args.estimator),
            num_queries=int(args.num_queries),
            perturbation_sides=str(args.perturbation_sides),
            query_microbatch_size=int(args.query_microbatch_size),
            multi_query_direction_mode=str(args.multi_query_direction_mode),
            population_size=int(args.population_size),
            sigma=float(args.eps if args.sigma is None else args.sigma),
            reward_shaping=str(args.reward_shaping),
            direction_provider=str(args.direction_provider),
            lozo_provider_mode=str(args.lozo_provider_mode),
            rank=int(args.rank),
            eps=float(args.eps),
            nu=int(args.nu),
            random_device=str(args.random_device),
            direction_sampling=str(args.direction_sampling),
            direction_scale=float(args.direction_scale),
            perturbation_normalization=str(args.perturbation_normalization),
            v_normalization=str(args.v_normalization),
            u_dim=args.u_dim,
            max_logits_tokens=int(args.max_logits_tokens),
            loss_impl="logprobs",
            seed=int(args.seed),
        ),
        param_metadata=weight_sync.get_hf_param_metadata(include_embeddings=False),
        update_state=update_state,
        sync_weight_update=False,
        weight_update_precision=str(args.weight_update_precision),
        qkv_update_mode=str(args.qkv_weight_update),
    )
    return engine, zo_model, tokenizer


def _build_update_state(
    args: argparse.Namespace,
    *,
    engine: ZOVLLMEngine,
    weight_sync: WeightSync,
):
    if str(args.direct_update_mode) != "accumulate":
        return None
    return AccumulatedLowRankUpdateState(
        weight_sync=weight_sync,
        engine=engine,
        precision=str(args.weight_update_precision),
        sync_device=False,
        qkv_update_mode=str(args.qkv_weight_update),
        u_beta=float(args.u_beta),
        u_norm_cap=args.u_norm_cap,
        gradient_accumulation_update_steps=int(args.gradient_accumulation_update_steps),
    )


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, int]:
    llm_kwargs = {}
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
    if int(args.max_num_seqs) > 0:
        llm_kwargs["max_num_seqs"] = int(args.max_num_seqs)
    return llm_kwargs


def _build_training_inputs(
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[Any, Any, type[ZOTrainer]]:
    if str(args.task_objective) == "registered":
        task = get_task(str(args.task))
        cfg = TaskConfig(
            name=task.name,
            num_train=int(args.num_train),
            num_dev=int(args.num_dev),
            num_eval=0,
            data_seed=_data_seed(args),
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
        )
        raw = task.load_splits(cfg).train
        rows = [dict(raw[index]) for index in range(len(raw))]
        dataset = build_objective_hf_dataset(
            rows * int(args.dataset_repeats),
            tokenizer,
            objective_name=task.vllm_train_objective,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
        )
        return dataset, VLLMDataCollator(), ZOTrainer
    if str(args.task_objective) == "sst2_classification":
        dataset = _build_sst2_classification_dataset(tokenizer, args)
        collator = VLLMDataCollator()
        return (
            dataset,
            collator,
            ZOTrainer,
        )
    dataset = _build_target_lm_dataset(tokenizer, args)
    data_collator = VLLMDataCollator()
    return (
        dataset,
        data_collator,
        ZOTrainer,
    )


def _build_sst2_classification_dataset(
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dataset:
    raw_split = load_dataset("glue", "sst2")["train"]
    selected = raw_split.shuffle(seed=_data_seed(args)).select(
        range(min(int(args.num_train), len(raw_split)))
    )
    if int(args.dataset_repeats) > 1:
        rows = [dict(selected[index]) for index in range(len(selected))]
        raw = Dataset.from_list(rows * int(args.dataset_repeats))
    else:
        raw = selected
    return raw.map(
        build_sst2_prompt_classification_preprocess(
            tokenizer,
            max_length=int(args.max_length),
        ),
        batched=True,
        remove_columns=raw.column_names,
    )


def _build_target_lm_dataset(tokenizer: Any, args: argparse.Namespace) -> Dataset:
    if str(args.task) == "sst2":
        raw_split = load_dataset("glue", "sst2")["train"]
        selected = raw_split.shuffle(seed=_data_seed(args)).select(
            range(min(int(args.num_train), len(raw_split)))
        )
        rows = [
            _sst2_prompt_target(dict(selected[index])) for index in range(len(selected))
        ]
        if int(args.dataset_repeats) > 1:
            rows = rows * int(args.dataset_repeats)
        raw = Dataset.from_list(rows)
        return raw.map(
            build_target_lm_preprocess(tokenizer, max_length=int(args.max_length)),
            batched=True,
            remove_columns=raw.column_names,
        )
    rows = [
        {"prompt": prompt, "target": target}
        for _ in range(int(args.dataset_repeats))
        for prompt, target in DEFAULT_PROMPTS
    ]
    raw = Dataset.from_list(rows)
    return raw.map(
        build_target_lm_preprocess(tokenizer, max_length=int(args.max_length)),
        batched=True,
        remove_columns=raw.column_names,
    )


def _sst2_prompt_target(row: dict[str, Any]) -> dict[str, str]:
    target = " positive" if int(row["label"]) == 1 else " negative"
    return {"prompt": f"Review: {row['sentence']}\nSentiment:", "target": target}


def _build_result(
    *,
    args: argparse.Namespace,
    output_metrics: dict[str, Any],
    step_s: list[float],
    phase3_profile: dict[str, list[float | int]],
    step_records: list[dict[str, Any]],
    log_history: list[dict[str, Any]],
    wall_clock_s: float,
) -> dict[str, Any]:
    tail_count = min(int(args.tail_steps), len(step_s))
    tail_values = step_s[-tail_count:] if tail_count else []
    mean_step_s = _mean(step_s)
    tail_step_s = _mean(tail_values)
    result = {
        "config": _config_dict(args),
        "zo_runtime_config": _zo_runtime_config(args),
        "hf_train_metrics": output_metrics,
        "phase3_timing": {
            "source": "hf_callback_on_step_begin_to_end",
            "num_steps_recorded": len(step_s),
            "step_s_mean": mean_step_s,
            "step_s_min": min(step_s) if step_s else None,
            "step_s_max": max(step_s) if step_s else None,
            "tail_steps": tail_count,
            "tail_step_s_mean": tail_step_s,
            "tail_steps_per_second": None if not tail_step_s else 1.0 / tail_step_s,
            "wall_clock_s": float(wall_clock_s),
        },
        "logged_profile_summary": _logged_profile_summary(log_history),
        "last_log": log_history[-1] if log_history else {},
    }
    result["phase3_profile"] = {
        key: _series_summary(values, tail_steps=int(args.tail_steps))
        for key, values in phase3_profile.items()
    }
    result["phase3_step_records"] = step_records
    zo_losses = [
        float(row["zo_loss"])
        for row in log_history
        if isinstance(row.get("zo_loss"), (float, int))
    ]
    result.update(
        {
            "initial_loss": zo_losses[0] if zo_losses else None,
            "final_loss": zo_losses[-1] if zo_losses else None,
            "loss_change": (
                None if not zo_losses else float(zo_losses[-1] - zo_losses[0])
            ),
            "eval_losses": [],
            "eval_metrics": [],
            "final_accuracy": None,
            "timing": {
                "total_s": float(sum(step_s)),
                "timing_source": "hf_callback_step_sum",
            },
        }
    )
    return result


def _config_dict(args: argparse.Namespace) -> dict[str, Any]:
    keys = sorted(vars(args))
    return {key: getattr(args, key) for key in keys}


def _zo_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    direction_provider = str(args.direction_provider)
    lozo_provider_mode = str(args.lozo_provider_mode)
    estimator = str(args.estimator)
    lazy_v_path = (
        estimator == "single_direction_antithetic"
        and direction_provider == "lozo"
        and lozo_provider_mode == "fast"
    )
    return {
        "estimator": estimator,
        "num_queries": int(args.num_queries),
        "perturbation_sides": str(args.perturbation_sides).replace("-", "_"),
        "query_microbatch_size": int(args.query_microbatch_size),
        "multi_query_direction_mode": str(args.multi_query_direction_mode),
        "population_size": int(args.population_size),
        "sigma": float(args.eps if args.sigma is None else args.sigma),
        "reward_shaping": str(args.reward_shaping),
        "direction_provider": direction_provider,
        "lozo_provider_mode": lozo_provider_mode,
        "lazy_v_path": lazy_v_path,
        "lazy_v_path_reason": (
            "LOZOFastDirectionProvider samples Gaussian U/V lazily per step"
            if lazy_v_path
            else "not single_direction_antithetic + lozo + fast"
        ),
        "rank": int(args.rank),
        "nu": int(args.nu),
        "random_device": str(args.random_device),
        "seed_sampler": "provider_default",
        "seed_sampler_seed": int(args.seed),
        "direction_sampling": str(args.direction_sampling),
        "direction_scale": float(args.direction_scale),
        "perturbation_normalization": str(args.perturbation_normalization),
        "v_normalization": str(args.v_normalization),
        "u_dim": args.u_dim,
        "direct_update_mode": str(args.direct_update_mode),
        "weight_update_precision": str(args.weight_update_precision),
        "qkv_weight_update": str(args.qkv_weight_update),
        "gradient_accumulation_update_steps": int(
            args.gradient_accumulation_update_steps
        ),
        "u_beta": float(args.u_beta),
        "u_norm_cap": args.u_norm_cap,
    }


def _data_seed(args: argparse.Namespace) -> int:
    return int(args.seed if args.data_seed is None else args.data_seed)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _series_summary(values: list[float | int], *, tail_steps: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "tail_mean": None, "min": None, "max": None}
    tail_count = min(int(tail_steps), len(values))
    tail_values = values[-tail_count:] if tail_count else []
    numeric = [float(value) for value in values]
    numeric_tail = [float(value) for value in tail_values]
    return {
        "count": len(values),
        "mean": float(sum(numeric) / len(numeric)),
        "tail_mean": (
            None if not numeric_tail else float(sum(numeric_tail) / len(numeric_tail))
        ),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
    }


def _logged_profile_summary(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in log_history:
        for key, value in row.items():
            if not (
                key.startswith("zo_profile_")
                or key.startswith("zo_update_")
                or key == "zo_estimator_query_count"
            ):
                continue
            if isinstance(value, (float, int)):
                grouped.setdefault(key, []).append(float(value))
    return {
        key: {
            "count": len(values),
            "mean": _mean(values),
            "last": values[-1] if values else None,
        }
        for key, values in sorted(grouped.items())
    }


def _profile_from_log_history(
    log_history: list[dict[str, Any]],
    *,
    step_s: list[float] | None,
) -> tuple[dict[str, list[float | int]], list[dict[str, Any]]]:
    """Build Phase 3 per-step profiles from ordinary HF Trainer logs."""

    metric_keys = {
        "score_plus_minus_total_s": "zo_profile_score_plus_minus_total_s",
        "set_plus_minus_directions_s": "zo_profile_set_plus_minus_directions_s",
        "score_token_groups_s": "zo_profile_score_token_groups_s",
        "zo_stepper_total_s": "zo_profile_zo_stepper_total_s",
        "scorer_batch_unpack_s": "zo_profile_scorer_batch_unpack_s",
        "scorer_loss_dispatch_s": "zo_profile_scorer_loss_dispatch_s",
        "scorer_loss_to_host_s": "zo_profile_scorer_loss_to_host_s",
        "scorer_total_s": "zo_profile_scorer_total_s",
        "scorer_forward_unpack_s": "zo_profile_scorer_forward_unpack_s",
        "scorer_engine_call_s": "zo_profile_scorer_engine_call_s",
        "scorer_output_postprocess_s": "zo_profile_scorer_output_postprocess_s",
        "scorer_forward_total_s": "zo_profile_scorer_forward_total_s",
        "worker_cuda_model_forward_s": "zo_profile_worker_cuda_model_forward_s",
        "worker_cuda_loss_s": "zo_profile_worker_cuda_loss_s",
    }
    profile: dict[str, list[float | int]] = {key: [] for key in metric_keys}
    profile["update_accumulate_fold_s"] = []
    profile["hf_step_over_stepper_s"] = []
    records: list[dict[str, Any]] = []
    profile_rows_by_step: dict[int, dict[str, Any]] = {}
    for row in log_history:
        if "zo_profile_zo_stepper_total_s" not in row:
            continue
        runtime_step = int(row.get("zo_step", row.get("step", 0)))
        if runtime_step <= 0 or runtime_step in profile_rows_by_step:
            continue
        profile_rows_by_step[runtime_step] = row
    profile_rows = [profile_rows_by_step[key] for key in sorted(profile_rows_by_step)]
    if step_s is not None and len(step_s) != len(profile_rows):
        raise RuntimeError(
            "Phase 3 callback/profile step count mismatch: "
            f"callback={len(step_s)} profile={len(profile_rows)}"
        )
    for index, row in enumerate(profile_rows):
        record: dict[str, Any] = {
            "step": int(row.get("zo_step", row.get("step", index + 1)))
        }
        for output_key, log_key in metric_keys.items():
            if log_key not in row:
                continue
            value = float(row[log_key])
            profile[output_key].append(value)
            record[output_key] = value
        update_s = float(row.get("zo_update_accumulate_s", 0.0)) + float(
            row.get("zo_update_fold_s", 0.0)
        )
        profile["update_accumulate_fold_s"].append(update_s)
        record["update_accumulate_fold_s"] = update_s
        if step_s is not None:
            full_step_s = float(step_s[index])
            residual_s = full_step_s - record["zo_stepper_total_s"]
            profile["hf_step_over_stepper_s"].append(residual_s)
            record["step_s"] = full_step_s
            record["hf_step_over_stepper_s"] = residual_s
        records.append(record)
    return profile, records


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _cleanup_engine(engine: ZOVLLMEngine | None) -> None:
    if engine is not None:
        engine.cleanup()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
