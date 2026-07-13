"""Phase 4 classification numeric comparison between legacy and HF ZO trainers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from zo_trainer import (
    VLLMDataCollator,
    ZOTrainer,
    ZOTrainerArguments,
    ZOVLLMCheckpointHandler,
)
from zo_vllm.config import VLLMZOConfig, ZOVLLMEngineConfig
from zo_vllm.core.lora_scope import resolve_lora_target_modules
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.training import VLLMZOModel, build_tokenizer
from zo_vllm.training.arguments import ZOTrainingArguments
from zo_vllm.training.direction import TokenProbeBatch
from zo_vllm.training.objective_scoring import (
    build_objective_batch,
    score_clean_objective,
)
from zo_vllm.training.objective_router import eval_objective_metrics
from zo_vllm.training.task_batches import (
    SUPPORTED_OBJECTIVES,
    load_objective_rows,
    resolve_objective_name,
)
from zo_vllm.training.task_encoding import ZOTaskDataCollator, ZOTaskEncodingConfig
from zo_vllm.training.update_state import AccumulatedLowRankUpdateState
from zo_vllm.training.vllm_zo_trainer import VLLMZOTrainer
from zo_vllm.tasks.hf_preprocessing import build_objective_hf_dataset


DEFAULT_MODEL = "facebook/opt-125m"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT_OBJECTIVES = frozenset(
    objective for objective in SUPPORTED_OBJECTIVES if objective != "prompt_nll"
)


class _LegacyStepModel:
    def __init__(self, zo_model: VLLMZOModel) -> None:
        self.zo_model = zo_model

    def estimate(self, batch: TokenProbeBatch, *, step: int) -> Any:
        return self.zo_model.estimate(batch, step=step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--task-objective",
        choices=sorted(ALIGNMENT_OBJECTIVES),
        default="sst2_classification",
    )
    parser.add_argument("--output-root", default="phase4/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-train", type=int, default=16)
    parser.add_argument("--num-dev", type=int, default=8)
    parser.add_argument("--num-eval", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--nu", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-logits-tokens", type=int, default=4096)
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
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument(
        "--save-strategy", choices=["no", "steps", "best"], default="no"
    )
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--checkpoint-mode",
        choices=["metadata", "native", "lora"],
        default="metadata",
    )
    parser.add_argument("--load-best-model-at-end", choices=["0", "1"], default="0")
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument("--greater-is-better", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--probe-atol", type=float, default=5e-3)
    parser.add_argument(
        "--require-accuracy-match",
        choices=["0", "1"],
        default="0",
        help="When set, accuracy mismatches fail the alignment check.",
    )
    parser.add_argument("--mode", choices=["both", "legacy", "hf"], default="both")
    args = parser.parse_args()
    args.task_objective = resolve_objective_name(str(args.task_objective))
    return args


def main() -> None:
    args = parse_args()
    _prepare_environment()
    run_id = args.run_id or (
        f"hf_trainer_{_safe_name(args.task_objective)}_alignment_"
        f"{_safe_name(args.model)}_s{args.steps}_b{args.batch_size}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "both":
        print(
            "[phase4-align] parameters="
            + json.dumps(_config_dict(args), sort_keys=True),
            flush=True,
        )
        _run_child(args, "legacy", run_id=run_id)
        _run_child(args, "hf", run_id=run_id)
        legacy = json.loads((output_dir / "legacy_result.json").read_text())
        hf = json.loads((output_dir / "hf_result.json").read_text())
        comparison = _compare(
            legacy,
            hf,
            atol=float(args.atol),
            rtol=float(args.rtol),
            probe_atol=float(args.probe_atol),
            require_accuracy_match=str(args.require_accuracy_match) == "1",
        )
        result = {
            "config": _config_dict(args),
            "legacy": legacy,
            "hf": hf,
            "comparison": comparison,
            "ok": bool(comparison["ok"]),
        }
        result_path = output_dir / "alignment_result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("[phase4-align] result=" + json.dumps(result, sort_keys=True), flush=True)
        print(f"[phase4-align] wrote {result_path}", flush=True)
        if not comparison["ok"]:
            raise SystemExit(1)
        return

    tokenizer = build_tokenizer(args.model, use_fast=False)
    train_rows, dev_rows, eval_rows = _load_rows(args)
    print(
        "[phase4-align] parameters=" + json.dumps(_config_dict(args), sort_keys=True),
        flush=True,
    )
    if args.mode == "legacy":
        result = _run_legacy(
            args, tokenizer, train_rows, dev_rows, eval_rows, output_dir
        )
        result_path = output_dir / "legacy_result.json"
    else:
        result = _run_hf(args, tokenizer, train_rows, dev_rows, eval_rows, output_dir)
        result_path = output_dir / "hf_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[phase4-align] wrote {result_path}", flush=True)


def _run_child(args: argparse.Namespace, mode: str, *, run_id: str) -> None:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
    ]
    for key, value in _config_dict(args).items():
        if key == "mode":
            continue
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if key == "run_id":
            value = run_id
        cmd.extend([flag, str(value)])
    cmd.extend(["--mode", mode])
    print("[phase4-align] child=" + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _run_legacy(
    args: argparse.Namespace,
    tokenizer: Any,
    train_rows: list[Any],
    dev_rows: list[Any],
    eval_rows: list[Any],
    output_dir: Path,
) -> dict[str, Any]:
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model = _build_zo_model(args)
        collator = ZOTaskDataCollator(
            tokenizer=tokenizer,
            config=ZOTaskEncodingConfig(
                objective_name=str(args.task_objective),
                max_length=int(args.max_length),
                max_new_tokens=int(args.max_new_tokens),
            ),
        )
        eval_history: list[dict[str, float | int]] = []

        def eval_fn() -> dict[str, float | int]:
            step = int(legacy_trainer.state.global_step)
            row = _score_eval(engine, tokenizer, dev_rows, eval_rows, args, step=step)
            eval_history.append(row)
            return {
                "eval_loss": row["loss"],
                "eval_accuracy": row["accuracy"],
            }

        legacy_trainer = VLLMZOTrainer(
            model=_LegacyStepModel(zo_model),
            args=ZOTrainingArguments(
                output_dir=str(output_dir / "legacy_output"),
                max_steps=int(args.steps),
                warmup_steps=0,
                per_device_train_batch_size=int(args.batch_size),
                logging_steps=max(1, int(args.eval_steps)),
                eval_steps=max(1, int(args.eval_steps)),
                save_strategy="no",
                save_steps=0,
                learning_rate=float(args.lr),
                seed=int(args.seed),
                dataloader_drop_last=False,
            ),
            train_dataset=train_rows,
            data_collator=collator,
            eval_fn=eval_fn,
        )
        initial = _score_eval(engine, tokenizer, dev_rows, eval_rows, args, step=0)
        started = time.time()
        output = legacy_trainer.train()
        return {
            "initial": initial,
            "eval_history": eval_history,
            "train_output": {
                "global_step": int(output.global_step),
                "metrics": dict(output.metrics),
            },
            "log_history": legacy_trainer.state.log_history,
            "wall_clock_s": float(time.time() - started),
        }
    finally:
        _cleanup_engine(engine)


def _run_hf(
    args: argparse.Namespace,
    tokenizer: Any,
    train_rows: list[Any],
    dev_rows: list[Any],
    eval_rows: list[Any],
    output_dir: Path,
) -> dict[str, Any]:
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model = _build_zo_model(args)
        train_dataset = build_objective_hf_dataset(
            train_rows,
            tokenizer,
            objective_name=str(args.task_objective),
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
        )
        eval_dataset = build_objective_hf_dataset(
            dev_rows,
            tokenizer,
            objective_name=str(args.task_objective),
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
        )
        model: Any = zo_model
        data_collator = VLLMDataCollator()
        checkpoint_handler = ZOVLLMCheckpointHandler(
            checkpoint_mode=str(args.checkpoint_mode),
            llm=engine.llm,
            weight_sync=zo_model.weight_sync,
            update_state=zo_model.update_state,
            precision=str(args.weight_update_precision),
            runtime_manifest={
                "model_name": str(args.model),
                "rank": int(args.rank),
                "task_objective": str(args.task_objective),
            },
        )
        greater_is_better = (
            None
            if str(args.greater_is_better) == "auto"
            else str(args.greater_is_better) == "1"
        )
        trainer = ZOTrainer(
            model=model,
            args=ZOTrainerArguments(
                output_dir=str(output_dir / "hf_output"),
                max_steps=int(args.steps),
                per_device_train_batch_size=int(args.batch_size),
                per_device_eval_batch_size=int(args.batch_size),
                logging_steps=max(1, int(args.eval_steps)),
                eval_strategy="steps",
                eval_steps=max(1, int(args.eval_steps)),
                save_strategy=str(args.save_strategy),
                save_steps=max(1, int(args.save_steps or args.eval_steps)),
                save_total_limit=int(args.save_total_limit),
                zo_checkpoint_mode=str(args.checkpoint_mode),
                load_best_model_at_end=str(args.load_best_model_at_end) == "1",
                metric_for_best_model=str(args.metric_for_best_model),
                greater_is_better=greater_is_better,
                disable_tqdm=True,
                report_to=[],
                remove_unused_columns=False,
                learning_rate=float(args.lr),
                lr_scheduler_type=str(args.lr_scheduler_type),
                seed=int(args.seed),
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=_compute_accuracy,
            checkpoint_handler=checkpoint_handler,
        )
        initial_metrics = trainer.evaluate(metric_key_prefix="initial_eval")
        initial_task_metrics = _score_eval(
            engine,
            tokenizer,
            dev_rows,
            eval_rows,
            args,
            step=0,
            update_state=zo_model.update_state,
        )
        started = time.time()
        output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        final_valid = _score_eval(
            engine,
            tokenizer,
            dev_rows,
            eval_rows,
            args,
            step=int(args.steps),
            update_state=zo_model.update_state,
        )
        eval_history = _hf_eval_history(trainer.state.log_history)
        final_loss = (
            float(eval_history[-1]["loss"])
            if eval_history
            else float(initial_metrics["initial_eval_loss"])
        )
        return {
            "initial": {
                "step": 0,
                "loss": float(initial_metrics["initial_eval_loss"]),
                "accuracy": float(initial_task_metrics["accuracy"]),
                "valid_accuracy": float(initial_task_metrics["valid_accuracy"]),
                "primary_metric": initial_task_metrics["primary_metric"],
                "dev_metrics": initial_task_metrics["dev_metrics"],
                "valid_metrics": initial_task_metrics["valid_metrics"],
            },
            "eval_history": eval_history,
            "initial_loss": float(initial_metrics["initial_eval_loss"]),
            "final_loss": final_loss,
            "loss_change": final_loss - float(initial_metrics["initial_eval_loss"]),
            "eval_losses": eval_history,
            "eval_metrics": eval_history,
            "final_accuracy": float(final_valid["valid_accuracy"]),
            "timing": {
                "total_s": float(output.metrics["train_runtime"]),
                "timing_source": "hf_train_runtime",
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "loaded_best_checkpoint": (
                trainer.state.best_model_checkpoint
                if str(args.load_best_model_at_end) == "1"
                else None
            ),
            "train_output": {
                "global_step": int(output.global_step),
                "metrics": dict(output.metrics),
            },
            "log_history": trainer.state.log_history,
            "final_valid": final_valid,
            "wall_clock_s": float(time.time() - started),
        }
    finally:
        _cleanup_engine(engine)


def _build_zo_model(args: argparse.Namespace) -> tuple[ZOVLLMEngine, VLLMZOModel]:
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(args.model)
    engine = ZOVLLMEngine(
        model=args.model,
        rank=int(args.rank),
        model_config=model_config,
        config=ZOVLLMEngineConfig(
            max_model_len=int(args.max_model_len),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            enforce_eager=bool(int(args.enforce_eager)),
            lora_rank=int(args.rank),
            target_modules=resolve_lora_target_modules(None),
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
    update_state = _build_update_state(args, engine=engine, weight_sync=weight_sync)
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
        update_state=update_state,
        sync_weight_update=False,
        weight_update_precision=str(args.weight_update_precision),
        qkv_update_mode=str(args.qkv_weight_update),
    )
    return engine, zo_model


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


def _score_eval(
    engine: ZOVLLMEngine,
    tokenizer: Any,
    dev_rows: list[Any],
    eval_rows: list[Any],
    args: argparse.Namespace,
    *,
    step: int,
    update_state: Any | None = None,
) -> dict[str, Any]:
    lora_id = None
    if update_state is not None:
        setter = getattr(update_state, "set_clean_lora_for_score", None)
        if callable(setter):
            lora_id, _ = setter(step=int(step))
    dev_batch = build_objective_batch(
        dev_rows,
        tokenizer,
        objective_name=str(args.task_objective),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )
    valid_batch = build_objective_batch(
        eval_rows,
        tokenizer,
        objective_name=str(args.task_objective),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
    )
    dev_score = score_clean_objective(
        engine,
        dev_batch,
        lora_id=lora_id,
        max_logits_tokens=int(args.max_logits_tokens),
        loss_impl="logprobs",
        score_chunk_size=0,
    )
    valid_score = score_clean_objective(
        engine,
        valid_batch,
        lora_id=lora_id,
        max_logits_tokens=int(args.max_logits_tokens),
        loss_impl="logprobs",
        score_chunk_size=0,
    )
    task_metrics = eval_objective_metrics(
        str(args.task_objective),
        engine.llm,
        tokenizer,
        dev_rows,
        eval_rows,
        max_logits_tokens=int(args.max_logits_tokens),
        loss_impl="logprobs",
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        lora_id=lora_id,
    )
    return {
        "step": int(step),
        "loss": float(dev_score.loss),
        "accuracy": float(
            dev_score.accuracy
            if task_metrics is None or task_metrics.dev_value is None
            else task_metrics.dev_value
        ),
        "valid_accuracy": float(
            valid_score.accuracy
            if task_metrics is None or task_metrics.valid_value is None
            else task_metrics.valid_value
        ),
        "primary_metric": (
            "accuracy" if task_metrics is None else task_metrics.primary_name
        ),
        "dev_metrics": None if task_metrics is None else task_metrics.dev_metrics,
        "valid_metrics": None if task_metrics is None else task_metrics.valid_metrics,
    }


def _compute_accuracy(prediction: Any) -> dict[str, float]:
    predictions = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    pred_ids = torch.as_tensor(predictions).argmax(dim=-1)
    labels = torch.as_tensor(prediction.label_ids).long()
    if labels.numel() == 0:
        return {"accuracy": 0.0}
    return {"accuracy": float((pred_ids.cpu() == labels.cpu()).float().mean().item())}


def _hf_eval_history(log_history: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    rows = []
    for row in log_history:
        if "eval_loss" not in row:
            continue
        rows.append(
            {
                "step": int(row.get("step", 0)),
                "loss": float(row["eval_loss"]),
                "accuracy": float(row.get("eval_accuracy", 0.0)),
            }
        )
    return rows


def _compare(
    legacy: dict[str, Any],
    hf: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    probe_atol: float,
    require_accuracy_match: bool,
) -> dict[str, Any]:
    checks = []
    checks.append(
        _check_close(
            "initial.loss", legacy["initial"]["loss"], hf["initial"]["loss"], atol, rtol
        )
    )
    checks.append(
        _check_close(
            "initial.accuracy",
            legacy["initial"]["accuracy"],
            hf["initial"]["accuracy"],
            atol,
            rtol,
            required=require_accuracy_match,
        )
    )
    legacy_curve = {int(row["step"]): row for row in legacy["eval_history"]}
    hf_curve = {int(row["step"]): row for row in hf["eval_history"]}
    for step in sorted(set(legacy_curve) & set(hf_curve)):
        checks.append(
            _check_close(
                f"step{step}.loss",
                legacy_curve[step]["loss"],
                hf_curve[step]["loss"],
                atol,
                rtol,
            )
        )
        checks.append(
            _check_close(
                f"step{step}.accuracy",
                legacy_curve[step]["accuracy"],
                hf_curve[step]["accuracy"],
                atol,
                rtol,
                required=require_accuracy_match,
            )
        )
    checks.append(
        {
            "name": "eval_steps_match",
            "ok": sorted(legacy_curve) == sorted(hf_curve),
            "legacy": sorted(legacy_curve),
            "hf": sorted(hf_curve),
        }
    )
    legacy_probes = {
        int(row["step"]): row
        for row in legacy["log_history"]
        if "loss_plus" in row and "loss_minus" in row
    }
    hf_probes = {
        int(row["zo_step"]): row
        for row in hf["log_history"]
        if "zo_loss_plus" in row and "zo_loss_minus" in row
    }
    common_probe_steps = sorted(set(legacy_probes) & set(hf_probes))
    checks.append(
        {
            "name": "first_probe_present",
            "ok": bool(common_probe_steps),
            "legacy": sorted(legacy_probes),
            "hf": sorted(hf_probes),
        }
    )
    if common_probe_steps:
        first_step = common_probe_steps[0]
        checks.append(
            _check_close(
                f"step{first_step}.loss_plus",
                legacy_probes[first_step]["loss_plus"],
                hf_probes[first_step]["zo_loss_plus"],
                probe_atol,
                rtol,
            )
        )
        checks.append(
            _check_close(
                f"step{first_step}.loss_minus",
                legacy_probes[first_step]["loss_minus"],
                hf_probes[first_step]["zo_loss_minus"],
                probe_atol,
                rtol,
            )
        )
    return {
        "ok": all(bool(item["ok"]) for item in checks),
        "checks": checks,
    }


def _check_close(
    name: str,
    left: Any,
    right: Any,
    atol: float,
    rtol: float,
    *,
    required: bool = True,
) -> dict[str, Any]:
    left_f = float(left)
    right_f = float(right)
    close = math.isclose(left_f, right_f, abs_tol=float(atol), rel_tol=float(rtol))
    return {
        "name": name,
        "ok": close if required else True,
        "required": bool(required),
        "value_ok": close,
        "legacy": left_f,
        "hf": right_f,
        "abs_diff": abs(left_f - right_f),
    }


def _load_rows(args: argparse.Namespace) -> tuple[list[Any], list[Any], list[Any]]:
    train_rows, dev_rows, eval_rows = load_objective_rows(
        str(args.task_objective),
        data_seed=int(args.data_seed),
        num_train=int(args.num_train),
        num_dev=int(args.num_dev),
        num_eval=int(args.num_eval),
    )
    return list(train_rows), list(dev_rows), list(eval_rows)


def _prepare_environment() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("WANDB_MODE", "offline")


def _config_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in sorted(vars(args))}


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
