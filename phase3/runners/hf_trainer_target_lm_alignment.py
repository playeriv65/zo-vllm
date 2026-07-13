"""Phase 3 numeric alignment for target-LM ZO training."""

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
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import SequentialSampler
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
from zo_vllm.training.arguments import ZOTrainingArguments
from zo_vllm.training.direction import TokenProbeBatch
from zo_vllm.training.vllm_zo_trainer import VLLMZOTrainer


DEFAULT_MODEL = "facebook/opt-125m"
DEFAULT_PHASE3_DATA_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_TASKS = ("synthetic", "sst2", "boolq", "copa")
DEFAULT_PROMPTS = (
    ("Review: good movie\nSentiment:", " positive"),
    ("Review: bad movie\nSentiment:", " negative"),
    ("Review: excellent acting\nSentiment:", " positive"),
    ("Review: boring story\nSentiment:", " negative"),
)


class _LegacyStepModel:
    def __init__(self, zo_model: VLLMZOModel) -> None:
        self.zo_model = zo_model

    def estimate(self, batch: TokenProbeBatch, *, step: int) -> Any:
        return self.zo_model.estimate(batch, step=step)


class _FeatureDataset(TorchDataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.rows[int(index)])


class _DirectionBatchCollator:
    def __call__(self, rows: list[dict[str, Any]]) -> TokenProbeBatch:
        token_groups = [[int(item) for item in row["input_ids"]] for row in rows]
        labels = [[int(item) for item in row["labels"]] for row in rows]
        return TokenProbeBatch(token_id_groups=token_groups, labels=labels)


class _FixedOrderZOTrainer(ZOTrainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for fixed-order alignment")
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.per_device_train_batch_size),
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="synthetic")
    parser.add_argument("--output-root", default="phase3/results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset-repeats", type=int, default=8)
    parser.add_argument("--num-train", type=int, default=16)
    parser.add_argument("--num-eval", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--nu", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_PHASE3_DATA_SEED)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="vLLM max_num_batched_tokens. Omit to use vLLM native auto selection.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-logits-tokens", type=int, default=4096)
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--mode", choices=["both", "legacy", "hf"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _prepare_environment()
    run_id = args.run_id or (
        "hf_trainer_target_lm_alignment_"
        f"{_safe_name(args.task)}_"
        f"{_safe_name(args.model)}_s{args.steps}_b{args.batch_size}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "both":
        print(
            "[phase3-align] parameters="
            + json.dumps(_config_dict(args), sort_keys=True),
            flush=True,
        )
        _run_child(args, "legacy", run_id=run_id)
        _run_child(args, "hf", run_id=run_id)
        legacy = json.loads((output_dir / "legacy_result.json").read_text())
        hf = json.loads((output_dir / "hf_result.json").read_text())
        comparison = _compare(legacy, hf, atol=float(args.atol), rtol=float(args.rtol))
        result = {
            "config": _config_dict(args),
            "legacy": legacy,
            "hf": hf,
            "comparison": comparison,
            "ok": bool(comparison["ok"]),
        }
        result_path = output_dir / "alignment_result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("[phase3-align] result=" + json.dumps(result, sort_keys=True), flush=True)
        print(f"[phase3-align] wrote {result_path}", flush=True)
        if not comparison["ok"]:
            raise SystemExit(1)
        return

    tokenizer = build_tokenizer(args.model, use_fast=False)
    train_rows, eval_rows = _build_feature_rows(tokenizer, args)
    print(
        "[phase3-align] parameters=" + json.dumps(_config_dict(args), sort_keys=True),
        flush=True,
    )
    if args.mode == "legacy":
        result = _run_legacy(args, train_rows, eval_rows, output_dir)
        result_path = output_dir / "legacy_result.json"
    else:
        result = _run_hf(args, tokenizer, train_rows, eval_rows, output_dir)
        result_path = output_dir / "hf_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[phase3-align] wrote {result_path}", flush=True)


def _run_child(args: argparse.Namespace, mode: str, *, run_id: str) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve())]
    for key, value in _config_dict(args).items():
        if key == "mode":
            continue
        if key == "run_id":
            value = run_id
        cmd.extend(["--" + key.replace("_", "-"), str(value)])
    cmd.extend(["--mode", mode])
    print("[phase3-align] child=" + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _run_legacy(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model = _build_zo_model(args)
        eval_history: list[dict[str, float | int]] = []

        def eval_fn() -> dict[str, float | int]:
            step = int(legacy_trainer.state.global_step)
            row = _score_eval(engine, eval_rows, args, step=step)
            eval_history.append(row)
            return {"eval_loss": row["loss"]}

        train_dataset = _FeatureDataset(train_rows)
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
            train_dataloader=DataLoader(
                train_dataset,
                batch_size=int(args.batch_size),
                sampler=SequentialSampler(train_dataset),
                collate_fn=_DirectionBatchCollator(),
                drop_last=False,
                num_workers=0,
            ),
            eval_fn=eval_fn,
        )
        initial = _score_eval(engine, eval_rows, args, step=0)
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
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    engine: ZOVLLMEngine | None = None
    try:
        engine, zo_model = _build_zo_model(args)
        collator = VLLMDataCollator()
        trainer = _FixedOrderZOTrainer(
            model=zo_model,
            args=ZOTrainerArguments(
                output_dir=str(output_dir / "hf_output"),
                max_steps=int(args.steps),
                per_device_train_batch_size=int(args.batch_size),
                per_device_eval_batch_size=int(args.batch_size),
                logging_steps=max(1, int(args.eval_steps)),
                eval_strategy="steps",
                eval_steps=max(1, int(args.eval_steps)),
                save_strategy="no",
                disable_tqdm=True,
                report_to=[],
                remove_unused_columns=False,
                learning_rate=float(args.lr),
                seed=int(args.seed),
            ),
            train_dataset=_FeatureDataset(train_rows),
            eval_dataset=_FeatureDataset(eval_rows),
            data_collator=collator,
            processing_class=tokenizer,
            compute_metrics=_compute_token_accuracy,
        )
        initial_metrics = trainer.evaluate(metric_key_prefix="initial_eval")
        started = time.time()
        output = trainer.train()
        return {
            "initial": {
                "step": 0,
                "loss": float(initial_metrics["initial_eval_loss"]),
                "accuracy": float(initial_metrics.get("initial_eval_accuracy", 0.0)),
            },
            "eval_history": _hf_eval_history(trainer.state.log_history),
            "train_output": {
                "global_step": int(output.global_step),
                "metrics": dict(output.metrics),
            },
            "log_history": trainer.state.log_history,
            "clean_final": _score_eval(engine, eval_rows, args, step=int(args.steps)),
            "wall_clock_s": float(time.time() - started),
        }
    finally:
        _cleanup_engine(engine)


def _build_zo_model(args: argparse.Namespace) -> tuple[ZOVLLMEngine, VLLMZOModel]:
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
                **(
                    {}
                    if args.max_num_batched_tokens is None
                    else {"max_num_batched_tokens": int(args.max_num_batched_tokens)}
                ),
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
    return engine, zo_model


def _build_feature_rows(
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_prompt_rows, eval_prompt_rows = _build_prompt_target_rows(args)
    train_raw = Dataset.from_list(train_prompt_rows)
    eval_raw = Dataset.from_list(eval_prompt_rows)
    train_mapped = train_raw.map(
        build_target_lm_preprocess(tokenizer, max_length=int(args.max_length)),
        batched=True,
        remove_columns=train_raw.column_names,
    )
    eval_mapped = eval_raw.map(
        build_target_lm_preprocess(tokenizer, max_length=int(args.max_length)),
        batched=True,
        remove_columns=eval_raw.column_names,
    )
    return (
        [dict(train_mapped[index]) for index in range(len(train_mapped))],
        [dict(eval_mapped[index]) for index in range(len(eval_mapped))],
    )


def _build_prompt_target_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    task = str(args.task)
    if task == "synthetic":
        rows = [
            {"prompt": prompt, "target": target}
            for _ in range(int(args.dataset_repeats))
            for prompt, target in DEFAULT_PROMPTS
        ]
        return rows, rows[: int(args.num_eval)]
    if task == "sst2":
        raw = load_dataset("glue", "sst2")
        return (
            _select_prompt_rows(
                raw["train"],
                args,
                converter=_sst2_prompt_target,
                limit=int(args.num_train),
            ),
            _select_prompt_rows(
                raw["validation"],
                args,
                converter=_sst2_prompt_target,
                limit=int(args.num_eval),
            ),
        )
    if task == "boolq":
        raw = load_dataset("super_glue", "boolq")
        return (
            _select_prompt_rows(
                raw["train"],
                args,
                converter=_boolq_prompt_target,
                limit=int(args.num_train),
            ),
            _select_prompt_rows(
                raw["validation"],
                args,
                converter=_boolq_prompt_target,
                limit=int(args.num_eval),
            ),
        )
    if task == "copa":
        raw = load_dataset("super_glue", "copa")
        return (
            _select_prompt_rows(
                raw["train"],
                args,
                converter=_copa_prompt_target,
                limit=int(args.num_train),
            ),
            _select_prompt_rows(
                raw["validation"],
                args,
                converter=_copa_prompt_target,
                limit=int(args.num_eval),
            ),
        )
    raise ValueError(f"unsupported task: {task}")


def _select_prompt_rows(
    dataset: Any,
    args: argparse.Namespace,
    *,
    converter,
    limit: int,
) -> list[dict[str, str]]:
    selected = dataset.shuffle(seed=_data_seed(args)).select(
        range(min(int(limit), len(dataset)))
    )
    rows = [converter(dict(selected[index])) for index in range(len(selected))]
    if int(args.dataset_repeats) > 1:
        rows = rows * int(args.dataset_repeats)
    return rows


def _sst2_prompt_target(row: dict[str, Any]) -> dict[str, str]:
    label = int(row["label"])
    target = " positive" if label == 1 else " negative"
    return {"prompt": f"Review: {row['sentence']}\nSentiment:", "target": target}


def _boolq_prompt_target(row: dict[str, Any]) -> dict[str, str]:
    label = int(row["label"])
    target = " yes" if label == 1 else " no"
    return {
        "prompt": f"Passage: {row['passage']}\nQuestion: {row['question']}?\nAnswer:",
        "target": target,
    }


def _copa_prompt_target(row: dict[str, Any]) -> dict[str, str]:
    question = str(row["question"]).strip().lower()
    relation = "cause" if question == "cause" else "effect"
    label = int(row["label"])
    choices = [str(row["choice1"]), str(row["choice2"])]
    target = " " + choices[label]
    prompt = (
        f"Premise: {row['premise']}\n"
        f"Which option is the more plausible {relation}?\n"
        f"A: {choices[0]}\nB: {choices[1]}\nAnswer:"
    )
    return {"prompt": prompt, "target": target}


def _score_eval(
    engine: ZOVLLMEngine,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    step: int,
) -> dict[str, float | int]:
    batch_losses: list[float] = []
    correct = 0.0
    total = 0
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        token_groups = [[int(item) for item in row["input_ids"]] for row in batch_rows]
        labels = [[int(item) for item in row["labels"]] for row in batch_rows]
        result = engine.forward_token_logits(
            token_groups,
            labels=labels,
            max_logits_tokens=int(args.max_logits_tokens),
            loss_impl="logprobs",
        )
        batch_losses.append(float(result.score.loss))
        target_labels = result.labels
        if target_labels.numel() == 0:
            continue
        predictions = result.logits.argmax(dim=-1).to(target_labels.device)
        correct += float((predictions == target_labels).float().sum().item())
        total += int(target_labels.numel())
    loss = sum(batch_losses) / float(len(batch_losses)) if batch_losses else 0.0
    accuracy = correct / float(total) if total else 0.0
    return {"step": int(step), "loss": float(loss), "accuracy": float(accuracy)}


def _compute_token_accuracy(prediction: Any) -> dict[str, float]:
    predictions = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    logits = torch.as_tensor(predictions)
    labels = torch.as_tensor(prediction.label_ids).long()
    if logits.numel() == 0 or labels.numel() == 0:
        return {"accuracy": 0.0}
    pred_ids = logits.argmax(dim=-1)
    if labels.ndim > pred_ids.ndim:
        labels = labels.squeeze()
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
) -> dict[str, Any]:
    checks = [
        _check_close(
            "initial.loss",
            legacy["initial"]["loss"],
            hf["initial"]["loss"],
            atol,
            rtol,
        )
    ]
    checks.append(
        _check_close(
            "initial.accuracy",
            legacy["initial"].get("accuracy", 0.0),
            hf["initial"].get("accuracy", 0.0),
            atol,
            rtol,
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
                legacy_curve[step].get("accuracy", 0.0),
                hf_curve[step].get("accuracy", 0.0),
                atol,
                rtol,
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
    return {"ok": all(bool(item["ok"]) for item in checks), "checks": checks}


def _check_close(
    name: str, left: Any, right: Any, atol: float, rtol: float
) -> dict[str, Any]:
    left_f = float(left)
    right_f = float(right)
    return {
        "name": name,
        "ok": math.isclose(left_f, right_f, abs_tol=float(atol), rel_tol=float(rtol)),
        "legacy": left_f,
        "hf": right_f,
        "abs_diff": abs(left_f - right_f),
    }


def _prepare_environment() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("WANDB_MODE", "offline")


def _config_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in sorted(vars(args))}


def _data_seed(args: argparse.Namespace) -> int:
    return int(args.data_seed)


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
