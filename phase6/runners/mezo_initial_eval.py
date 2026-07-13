#!/usr/bin/env python3
"""Compute official MeZO-style initial SST-2 eval loss without training."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOZO_ROOT = PROJECT_ROOT / "third_party" / "LOZO" / "large_models"
sys.path.insert(0, str(LOZO_ROOT))

from tasks import get_task  # noqa: E402
from utils import (  # noqa: E402
    DataCollatorWithPaddingAndNesting,
    encode_prompt,
    forward_wrap_with_option_len,
)
from zo_vllm.tasks.tokenization import (  # noqa: E402
    OPT_BOS_MODES,
    OPT_BOS_NATIVE,
    configure_opt_tokenizer,
    tokenizer_from_pretrained_kwargs,
)


LOGGER = logging.getLogger("phase6.mezo_initial_eval")


class ListDataset(Dataset):
    def __init__(self, data: list[Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="facebook/opt-13b")
    parser.add_argument("--task-name", default="SST2")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-dev", type=int, default=500)
    parser.add_argument("--num-eval", type=int, default=872)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-set-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--load-float16", action="store_true")
    parser.add_argument("--scope-label", default="full")
    parser.add_argument(
        "--opt-bos-mode",
        choices=list(OPT_BOS_MODES),
        default=OPT_BOS_NATIVE,
        help=(
            "OPT tokenizer BOS policy. native keeps HF default BOS=</s> id 2; "
            "lozo uses <s> id 0 for strict LOZO/MeZO reproduction."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args: argparse.Namespace):
    free_gb = (
        int(torch.cuda.mem_get_info()[0] / 1024**3) if torch.cuda.is_available() else 0
    )
    dtype = torch.float16 if args.load_float16 else torch.float32
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        device_map="auto",
        torch_dtype=dtype,
        max_memory=(
            {i: f"{max(free_gb - 5, 1)}GB" for i in range(torch.cuda.device_count())}
            if torch.cuda.is_available()
            else None
        ),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        **tokenizer_from_pretrained_kwargs(
            args.model_name,
            opt_bos_mode=args.opt_bos_mode,
        ),
    )
    configure_opt_tokenizer(
        tokenizer,
        args.model_name,
        opt_bos_mode=args.opt_bos_mode,
    )
    tokenizer.padding_side = "left"
    return model, tokenizer


def convert_samples(
    task: Any, tokenizer: Any, samples: list[Any], args: argparse.Namespace
) -> ListDataset:
    data: list[list[dict[str, Any]]] = []
    template = task.get_template()
    for sample in samples:
        encoded_candidates, option_lens = encode_prompt(
            task,
            template,
            [],
            sample,
            tokenizer,
            max_length=args.max_length,
            generation=task.generation,
            generation_with_gold=True,
            max_new_tokens=args.max_new_tokens,
        )
        if task.generation:
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct_candidate_id = sample.candidates.index(sample.correct_candidate)
        data.append(
            [
                {
                    "input_ids": encoded_candidates[i],
                    "labels": correct_candidate_id,
                    "option_len": option_lens[i],
                    "num_options": len(sample.candidates),
                }
                for i in range(len(encoded_candidates))
            ]
        )
    return ListDataset(data)


def compute_accuracy(
    model: Any, tokenizer: Any, task: Any, samples: list[Any], args: argparse.Namespace
) -> float:
    correct = 0
    template = task.get_template()
    with torch.inference_mode():
        for sample in samples:
            encoded_candidates, option_lens = encode_prompt(
                task,
                template,
                [],
                sample,
                tokenizer,
                max_length=args.max_length,
                generation=task.generation,
                max_new_tokens=args.max_new_tokens,
            )
            scores = []
            for input_ids, option_len in zip(
                encoded_candidates, option_lens, strict=True
            ):
                tensor = torch.tensor([input_ids], device=model.device)
                logits = model(input_ids=tensor).logits[0, :-1]
                labels = tensor[0, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                selected = log_probs[
                    torch.arange(len(labels), device=labels.device), labels
                ]
                scores.append(float(selected[-option_len:].mean().detach().cpu()))
            pred = int(np.argmax(scores))
            correct += int(sample.candidates[pred] == sample.correct_candidate)
    return correct / max(len(samples), 1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    set_seed(args.seed)
    start = time.time()

    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        seed=args.train_set_seed,
    )
    train_dev = train_sets[0]
    dev_samples = train_dev[-args.num_dev :]
    eval_samples = task.sample_subset(
        data_split="valid", seed=args.train_set_seed, num=args.num_eval
    )

    model, tokenizer = load_model_and_tokenizer(args)
    model.original_forward = model.forward
    model.forward = forward_wrap_with_option_len.__get__(model, type(model))

    eval_dataset = convert_samples(task, tokenizer, dev_samples, args)
    training_args = TrainingArguments(
        output_dir=str(Path(args.output).with_suffix("")) + "_hf_tmp",
        per_device_eval_batch_size=args.eval_batch_size,
        per_device_train_batch_size=args.batch_size,
        report_to=[],
        fp16=False,
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPaddingAndNesting(
            tokenizer, pad_to_multiple_of=8
        ),
    )
    eval_metrics = trainer.evaluate()
    dev_accuracy = compute_accuracy(model, tokenizer, task, dev_samples, args)
    valid_accuracy = compute_accuracy(model, tokenizer, task, eval_samples, args)

    result = {
        "model_name": args.model_name,
        "task_name": args.task_name,
        "scope_label": args.scope_label,
        "seed": args.seed,
        "train_set_seed": args.train_set_seed,
        "num_train": args.num_train,
        "num_dev": args.num_dev,
        "num_eval": args.num_eval,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "initial_dev_loss": float(eval_metrics["eval_loss"]),
        "initial_dev_accuracy": float(dev_accuracy),
        "initial_valid_accuracy": float(valid_accuracy),
        "runtime_s": time.time() - start,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    LOGGER.info(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
