"""Phase 4 HF Trainer semantic migration probe.

This script keeps Phase 4 validation outside the generic ``zo_trainer`` package.
It runs a medium-length HF-native ZOTrainer job and records clean eval loss,
script-local eval accuracy, and ZO probe/update logs from Hugging Face history.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoConfig

from zo_trainer import (
    VLLMDataCollator,
    ZOTrainer,
    ZOTrainerArguments,
    build_target_lm_preprocess,
)
from zo_vllm.config import VLLMZOConfig, ZOVLLMEngineConfig
from zo_vllm.core.lora_scope import resolve_lora_target_modules
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.training import VLLMZOModel, build_tokenizer


DEFAULT_MODEL = "facebook/opt-125m"
TRAIN_EXAMPLES = (
    ("Review: good movie\nSentiment:", " positive"),
    ("Review: bad movie\nSentiment:", " negative"),
    ("Review: excellent acting\nSentiment:", " positive"),
    ("Review: boring story\nSentiment:", " negative"),
    ("Review: wonderful plot\nSentiment:", " positive"),
    ("Review: terrible ending\nSentiment:", " negative"),
)
EVAL_EXAMPLES = (
    ("Review: great film\nSentiment:", " positive"),
    ("Review: awful film\nSentiment:", " negative"),
    ("Review: enjoyable cast\nSentiment:", " positive"),
    ("Review: dull cast\nSentiment:", " negative"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-root", default="phase4/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--dataset-repeats", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--nu", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-logits-tokens", type=int, default=1024)
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.eval_steps <= 0:
        raise SystemExit("--eval-steps must be positive")
    if args.dataset_repeats <= 0:
        raise SystemExit("--dataset-repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    _prepare_environment()
    run_id = args.run_id or (
        "hf_trainer_semantic_"
        f"{_safe_name(args.model)}_s{args.steps}_b{args.batch_size}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    engine: ZOVLLMEngine | None = None
    started = time.time()
    try:
        engine, zo_model, tokenizer = _build_zo_model(args)
        train_dataset, eval_dataset = _build_datasets(tokenizer, args)
        trainer = ZOTrainer(
            model=zo_model,
            args=ZOTrainerArguments(
                output_dir=str(output_dir / "hf_output"),
                max_steps=int(args.steps),
                per_device_train_batch_size=int(args.batch_size),
                per_device_eval_batch_size=int(args.eval_batch_size),
                logging_steps=max(1, int(args.logging_steps)),
                eval_strategy="steps",
                eval_steps=max(1, int(args.eval_steps)),
                save_strategy="no",
                disable_tqdm=True,
                report_to=[],
                remove_unused_columns=False,
                learning_rate=float(args.lr),
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VLLMDataCollator(),
            processing_class=tokenizer,
            compute_metrics=compute_token_accuracy,
        )
        print(
            "[phase4-hf] parameters=" + json.dumps(_config_dict(args), sort_keys=True),
            flush=True,
        )
        initial_metrics = trainer.evaluate(metric_key_prefix="initial_eval")
        train_output = trainer.train()
        final_metrics = trainer.evaluate(metric_key_prefix="final_eval")
        result = _build_result(
            args=args,
            initial_metrics=initial_metrics,
            train_metrics=train_output.metrics,
            final_metrics=final_metrics,
            log_history=trainer.state.log_history,
            wall_clock_s=time.time() - started,
        )
        result_path = output_dir / "phase4_hf_semantic_result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("[phase4-hf] result=" + json.dumps(result, sort_keys=True), flush=True)
        print(f"[phase4-hf] wrote {result_path}", flush=True)
    finally:
        _cleanup_engine(engine)


def compute_token_accuracy(prediction: Any) -> dict[str, float]:
    predictions = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    labels = prediction.label_ids
    logits = np.asarray(predictions)
    label_ids = np.asarray(labels)
    if logits.size == 0 or label_ids.size == 0:
        return {"accuracy": 0.0}
    pred_ids = np.argmax(logits, axis=-1)
    mask = label_ids != -100
    if mask.ndim > pred_ids.ndim:
        mask = np.squeeze(mask)
    pred_flat = np.asarray(pred_ids)[mask].reshape(-1)
    label_flat = np.asarray(label_ids)[mask].reshape(-1)
    if label_flat.size == 0:
        return {"accuracy": 0.0}
    return {"accuracy": float(np.mean(pred_flat == label_flat))}


def _prepare_environment() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("WANDB_MODE", "offline")


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
            llm_kwargs={
                "max_num_batched_tokens": int(args.max_num_batched_tokens),
                "max_num_seqs": int(args.max_num_seqs),
            },
        ),
    )
    weight_sync = WeightSync(
        engine.llm,
        num_layers=model_config.num_hidden_layers,
        model_config=model_config,
    )
    zo_model = VLLMZOModel(
        engine=engine,
        weight_sync=weight_sync,
        config=VLLMZOConfig(
            direction_provider="lozo",
            lozo_provider_mode="fast",
            rank=int(args.rank),
            eps=float(args.eps),
            nu=int(args.nu),
            random_device="cuda",
            direction_sampling="flat",
            direction_scale=1.0,
            perturbation_normalization="rms",
            max_logits_tokens=int(args.max_logits_tokens),
            loss_impl="logprobs",
            seed=int(args.seed),
        ),
        param_metadata=weight_sync.get_hf_param_metadata(include_embeddings=False),
        sync_weight_update=False,
        qkv_update_mode="batched",
    )
    return engine, zo_model, tokenizer


def _build_datasets(
    tokenizer: Any, args: argparse.Namespace
) -> tuple[Dataset, Dataset]:
    train_rows = [
        {"prompt": prompt, "target": target}
        for _ in range(int(args.dataset_repeats))
        for prompt, target in TRAIN_EXAMPLES
    ]
    eval_rows = [
        {"prompt": prompt, "target": target} for prompt, target in EVAL_EXAMPLES
    ]
    preprocess = build_target_lm_preprocess(tokenizer, max_length=int(args.max_length))
    train_raw = Dataset.from_list(train_rows)
    eval_raw = Dataset.from_list(eval_rows)
    return (
        train_raw.map(
            preprocess,
            batched=True,
            remove_columns=train_raw.column_names,
        ),
        eval_raw.map(
            preprocess,
            batched=True,
            remove_columns=eval_raw.column_names,
        ),
    )


def _build_result(
    *,
    args: argparse.Namespace,
    initial_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    log_history: list[dict[str, Any]],
    wall_clock_s: float,
) -> dict[str, Any]:
    eval_rows = [
        row
        for row in log_history
        if any(key.endswith("_loss") and key.startswith("eval") for key in row)
        or "eval_loss" in row
    ]
    probe_rows = [row for row in log_history if "zo_projected_grad" in row]
    initial_loss = _metric(initial_metrics, "initial_eval_loss")
    final_loss = _metric(final_metrics, "final_eval_loss")
    initial_accuracy = _metric(initial_metrics, "initial_eval_accuracy")
    final_accuracy = _metric(final_metrics, "final_eval_accuracy")
    checks = {
        "initial_eval_loss_finite": _finite(initial_loss),
        "final_eval_loss_finite": _finite(final_loss),
        "final_eval_accuracy_finite": _finite(final_accuracy),
        "periodic_eval_logged": bool(eval_rows),
        "zo_probe_metrics_logged": bool(probe_rows),
    }
    return {
        "config": _config_dict(args),
        "initial_metrics": initial_metrics,
        "train_metrics": train_metrics,
        "final_metrics": final_metrics,
        "semantic_summary": {
            "initial_eval_loss": initial_loss,
            "final_eval_loss": final_loss,
            "eval_loss_delta": None
            if initial_loss is None or final_loss is None
            else float(final_loss - initial_loss),
            "initial_eval_accuracy": initial_accuracy,
            "final_eval_accuracy": final_accuracy,
            "eval_accuracy_delta": None
            if initial_accuracy is None or final_accuracy is None
            else float(final_accuracy - initial_accuracy),
            "num_periodic_eval_rows": len(eval_rows),
            "num_probe_log_rows": len(probe_rows),
            "wall_clock_s": float(wall_clock_s),
        },
        "checks": checks,
        "ok": all(checks.values()),
        "last_logs": log_history[-5:],
    }


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _config_dict(args: argparse.Namespace) -> dict[str, Any]:
    keys = sorted(vars(args))
    return {key: getattr(args, key) for key in keys}


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
