#!/usr/bin/env python
"""Compare zo_vllm SuperGLUE prompts against the original LOZO source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOZO_ROOT = PROJECT_ROOT / "third_party" / "LOZO" / "large_models"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LOZO_ROOT))

from tasks import get_task as get_lozo_task  # type: ignore  # noqa: E402
from utils import encode_prompt, forward_wrap_with_option_len  # type: ignore  # noqa: E402

from zo_vllm.core.binary_option_objective import (  # noqa: E402
    classification_loss_from_multi_option_nll,
)
from zo_vllm.tasks import TaskConfig, get_task as get_zovllm_task  # noqa: E402
from zo_vllm.tasks.superglue import (  # noqa: E402
    dataset_to_superglue_rows,
    encode_superglue_option_prompts,
)
from zo_vllm.tasks.tokenization import configure_opt_tokenizer  # noqa: E402


TASKS = {
    "boolq": ("superglue_boolq", "BoolQ"),
    "cb": ("superglue_cb", "CB"),
    "copa": ("superglue_copa", "Copa"),
    "multirc": ("superglue_multirc", "MultiRC"),
    "rte": ("superglue_rte", "RTE"),
    "wic": ("superglue_wic", "WIC"),
    "wsc": ("superglue_wsc", "WSC"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS))
    parser.add_argument("--split", choices=["train", "dev", "eval"], default="dev")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-dev", type=int, default=500)
    parser.add_argument("--num-eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--include-inputs", action="store_true")
    parser.add_argument("--jsonl", default=None)
    return parser.parse_args()


def lozo_splits(task: Any, args: argparse.Namespace) -> tuple[list[Any], list[Any], list[Any]]:
    train_samples = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        seed=args.seed,
    )[0]
    dev_samples = train_samples[-args.num_dev :]
    train_samples = train_samples[: -args.num_dev]
    eval_samples = task.sample_subset(
        data_split="valid",
        seed=args.seed,
        num=args.num_eval,
    )
    return train_samples, dev_samples, eval_samples


def zo_rows(task_name: str, args: argparse.Namespace):
    adapter = get_zovllm_task(task_name)
    cfg = TaskConfig(
        name=task_name,
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        data_seed=args.seed,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    splits = adapter.load_splits(cfg)
    return (
        dataset_to_superglue_rows(splits.train, task_name=task_name),
        dataset_to_superglue_rows(splits.dev, task_name=task_name),
        dataset_to_superglue_rows(splits.eval, task_name=task_name),
    )


def select_split(split: str, train: list[Any], dev: list[Any], eval_rows: list[Any]) -> list[Any]:
    if split == "train":
        return train
    if split == "dev":
        return dev
    return eval_rows


def lozo_correct_index(sample: Any) -> int:
    if isinstance(sample.correct_candidate, list):
        return sample.candidates.index(sample.correct_candidate[0])
    return sample.candidates.index(sample.correct_candidate)


def load_model(args: argparse.Namespace):
    if args.model_name is None:
        return None
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    )
    model.to(args.device)
    model.eval()
    model.original_forward = model.forward
    return model


def left_pad(rows: list[list[int]], pad_id: int) -> torch.Tensor:
    width = max(len(row) for row in rows)
    padded = [[pad_id] * (width - len(row)) + row for row in rows]
    return torch.tensor(padded, dtype=torch.long)


def lozo_classification_loss(
    model: Any,
    input_ids: list[list[int]],
    option_lens: list[int],
    label: int,
    pad_id: int,
    device: str,
) -> float:
    batch = left_pad(input_ids, pad_id).to(device)
    labels = torch.tensor([label] * len(input_ids), dtype=torch.long, device=device)
    option_len_tensor = torch.tensor(option_lens, dtype=torch.long, device=device)
    num_options = torch.tensor(
        [len(input_ids)] * len(input_ids),
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        output = forward_wrap_with_option_len(
            model,
            input_ids=batch,
            labels=labels,
            option_len=option_len_tensor,
            num_options=num_options,
            return_dict=True,
        )
    return float(output.loss.detach().cpu())


def zo_option_mean_nll(
    model: Any,
    input_ids: list[list[int]],
    option_lens: list[int],
    pad_id: int,
    device: str,
) -> list[float]:
    batch = left_pad(input_ids, pad_id).to(device)
    with torch.inference_mode():
        logits = model.original_forward(input_ids=batch).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_labels == pad_id, -100)
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        option_mean_nll = []
        for row_index, option_len in enumerate(option_lens):
            labels = shift_labels[row_index]
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[-int(option_len) :] = labels[-int(option_len) :] != -100
            safe_labels = labels.masked_fill(~mask, 0)
            selected = log_probs[row_index].gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            option_mean_nll.append(float((-selected[mask]).mean().detach().cpu()))
    return option_mean_nll


def zo_formula_loss(
    model: Any,
    input_ids: list[list[int]],
    option_lens: list[int],
    label: int,
    pad_id: int,
    device: str,
) -> tuple[float, list[float]]:
    option_mean_nll = zo_option_mean_nll(
        model,
        input_ids,
        option_lens,
        pad_id,
        device,
    )
    return (
        classification_loss_from_multi_option_nll([option_mean_nll], [label]),
        option_mean_nll,
    )


def compare_task(
    task_key: str,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    zo_name, lozo_name = TASKS[task_key]
    lozo_task = get_lozo_task(lozo_name)
    lozo_train, lozo_dev, lozo_eval = lozo_splits(lozo_task, args)
    zo_train, zo_dev, zo_eval = zo_rows(zo_name, args)
    lozo_rows = select_split(args.split, lozo_train, lozo_dev, lozo_eval)
    zo_task_rows = select_split(args.split, zo_train, zo_dev, zo_eval)
    count = min(args.limit, len(lozo_rows), len(zo_task_rows))
    records: list[dict[str, Any]] = []
    for index in range(count):
        lozo_sample = lozo_rows[index]
        zo_row = zo_task_rows[index]
        lozo_ids, lozo_lens = encode_prompt(
            lozo_task,
            lozo_task.get_template(),
            [],
            lozo_sample,
            tokenizer,
            max_length=args.max_length,
            generation=lozo_task.generation,
            max_new_tokens=args.max_new_tokens,
        )
        zo_encoded = encode_superglue_option_prompts([zo_row], tokenizer)
        zo_ids = zo_encoded.option_ids[0]
        zo_lens = zo_encoded.option_lens[0]
        label = lozo_correct_index(lozo_sample)
        record = {
            "task": task_key,
            "split": args.split,
            "index": index,
            "lozo_id": lozo_sample.id,
            "zo_id": zo_row.idx,
            "lozo_candidates": list(lozo_sample.candidates),
            "zo_candidates": list(zo_row.candidates),
            "lozo_label": label,
            "zo_label": zo_row.label,
            "input_ids_match": lozo_ids == zo_ids,
            "option_lens_match": lozo_lens == zo_lens,
            "labels_match": label == zo_row.label,
            "lozo_option_lens": lozo_lens,
            "zo_option_lens": zo_lens,
        }
        if args.include_inputs:
            record["lozo_input_ids"] = lozo_ids
            record["zo_input_ids"] = zo_ids
            record["lozo_texts"] = [tokenizer.decode(ids) for ids in lozo_ids]
            record["zo_texts"] = [tokenizer.decode(ids) for ids in zo_ids]
        if not record["input_ids_match"]:
            mismatches = [
                option_index
                for option_index, (left, right) in enumerate(zip(lozo_ids, zo_ids))
                if left != right
            ]
            record["mismatched_options"] = mismatches
            if mismatches:
                option_index = mismatches[0]
                record["lozo_text"] = tokenizer.decode(lozo_ids[option_index])
                record["zo_text"] = tokenizer.decode(zo_ids[option_index])
                record["lozo_ids"] = lozo_ids[option_index]
                record["zo_ids"] = zo_ids[option_index]
        if model is not None:
            pad_id = tokenizer.pad_token_id
            record["lozo_loss"] = lozo_classification_loss(
                model,
                lozo_ids,
                lozo_lens,
                label,
                pad_id,
                args.device,
            )
            record["zo_formula_loss"], record["zo_option_mean_nll"] = zo_formula_loss(
                model,
                zo_ids,
                zo_lens,
                zo_row.label,
                pad_id,
                args.device,
            )
            record["loss_abs_diff"] = abs(record["lozo_loss"] - record["zo_formula_loss"])
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    tokenizer_name = args.model_name or "facebook/opt-13b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    configure_opt_tokenizer(tokenizer, tokenizer_name)
    if "opt" in tokenizer_name.lower():
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = load_model(args)
    output = open(args.jsonl, "w") if args.jsonl else None
    try:
        all_records: list[dict[str, Any]] = []
        for task_key in args.tasks:
            records = compare_task(task_key, tokenizer, model, args)
            all_records.extend(records)
            mismatches = [
                row
                for row in records
                if not (
                    row["input_ids_match"]
                    and row["option_lens_match"]
                    and row["labels_match"]
                )
            ]
            max_loss_diff = max(
                (row.get("loss_abs_diff", 0.0) for row in records),
                default=0.0,
            )
            print(
                f"{task_key}: rows={len(records)} mismatches={len(mismatches)} "
                f"max_loss_diff={max_loss_diff:.6e}",
                flush=True,
            )
            if mismatches:
                first = mismatches[0]
                print(json.dumps(first, ensure_ascii=False, indent=2)[:4000], flush=True)
            if output is not None:
                for row in records:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        total_bad = sum(
            not (
                row["input_ids_match"]
                and row["option_lens_match"]
                and row["labels_match"]
            )
            for row in all_records
        )
        if total_bad:
            raise SystemExit(1)
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
