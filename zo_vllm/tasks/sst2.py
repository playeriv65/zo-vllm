"""SST-2 task adapter and prompt encoding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

from .base import TaskConfig, TaskSplits, build_task_data_collator, shuffled_select


@dataclass(frozen=True)
class SST2Row:
    sentence: str
    label: int
    idx: int | None = None


class SST2ClassificationDataset(TorchDataset):
    def __init__(self, rows: Iterable[SST2Row], tokenizer: PreTrainedTokenizerBase):
        self.items = [encode_sst2_classification_item(row, tokenizer) for row in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class SST2TaskAdapter:
    name = "sst2"
    official_task_name = "SST2"
    vllm_train_objective = "sst2_classification"
    metric_names = ("eval_loss", "eval_accuracy")
    primary_metric = "eval_loss"
    greater_is_better = False
    supports_official_lozo = True

    def load_splits(self, cfg: TaskConfig) -> TaskSplits:
        raw = load_dataset("glue", "sst2")
        train_dev = shuffled_select(
            raw["train"],
            seed=cfg.data_seed,
            num=int(cfg.num_train) + int(cfg.num_dev),
            shuffle_impl=cfg.shuffle_impl,
        )
        train = train_dev.select(range(min(int(cfg.num_train), len(train_dev))))
        dev_start = min(int(cfg.num_train), len(train_dev))
        dev_end = min(dev_start + int(cfg.num_dev), len(train_dev))
        dev = (
            train_dev.select(range(dev_start, dev_end))
            if dev_end > dev_start
            else train_dev.select([])
        )
        eval_split = shuffled_select(
            raw["validation"],
            seed=cfg.data_seed,
            num=cfg.num_eval,
            shuffle_impl=cfg.shuffle_impl,
        )
        return TaskSplits(train=train, dev=dev, eval=eval_split)

    def official_lozo_args(self, cfg: TaskConfig) -> list[str]:
        return ["--train_as_classification"]

    def vllm_args(self, cfg: TaskConfig) -> list[str]:
        return ["--train-objective", self.vllm_train_objective]

    def data_collator(self, tokenizer: Any, cfg: TaskConfig | None = None):
        return build_task_data_collator(
            tokenizer,
            objective_name=self.vllm_train_objective,
            row_converter=dataset_to_sst2_rows,
            cfg=cfg,
        )


def sst2_stem(row: SST2Row) -> str:
    return f"{row.sentence.strip()} It was"


def sst2_verbalized(row: SST2Row, candidate: int) -> str:
    verbalizer = {0: "terrible", 1: "great"}
    return f"{sst2_stem(row)} {verbalizer[candidate]}"


def dataset_to_sst2_rows(dataset: HFDataset) -> list[SST2Row]:
    """Convert HF Dataset rows into SST-2 scoring rows."""

    return [
        SST2Row(
            sentence=str(item["sentence"]),
            label=int(item["label"]),
            idx=int(item["idx"]) if "idx" in item and item["idx"] is not None else None,
        )
        for item in dataset
    ]


def load_sst2_rows(split: str) -> list[SST2Row]:
    dataset = load_dataset("glue", "sst2", split=split)
    return dataset_to_sst2_rows(dataset)


def sample_sst2_rows(rows: list[SST2Row], seed: int, num: int | None) -> list[SST2Row]:
    with np_random_seed(seed):
        indices = np.random.permutation(len(rows)).tolist()
    if num is not None:
        indices = indices[:num]
    return [rows[i] for i in indices]


def sample_sst2_train_dev(
    seed: int,
    num_train: int,
    num_dev: int,
) -> tuple[list[SST2Row], list[SST2Row]]:
    rows = load_sst2_rows("train")
    sampled = sample_sst2_rows(rows, seed, num_train + num_dev)
    return sampled[:num_train], sampled[num_train : num_train + num_dev]


def sample_sst2_validation(seed: int, num_eval: int | None) -> list[SST2Row]:
    return sample_sst2_rows(load_sst2_rows("validation"), seed, num_eval)


def encode_sst2_classification_item(
    row: SST2Row,
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict]:
    stem_len = len(tokenizer.encode(sst2_stem(row)))
    item = []
    for candidate in (0, 1):
        input_ids = tokenizer.encode(sst2_verbalized(row, candidate))
        item.append(
            {
                "input_ids": input_ids,
                "labels": int(row.label),
                "option_len": len(input_ids) - stem_len,
                "num_options": 2,
            }
        )
    return item


def encode_sst2_vllm_prompts(
    rows: list[SST2Row],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[int]]:
    stems = [tokenizer.encode(sst2_stem(row), add_special_tokens=True) for row in rows]
    neg = [
        tokenizer.encode(sst2_verbalized(row, 0), add_special_tokens=True)
        for row in rows
    ]
    pos = [
        tokenizer.encode(sst2_verbalized(row, 1), add_special_tokens=True)
        for row in rows
    ]
    labels = [int(row.label) for row in rows]
    return stems, neg, pos, labels


def classification_loss_from_nll(
    stem_nll: list[float],
    neg_nll: list[float],
    pos_nll: list[float],
    labels: list[int],
) -> float:
    logits = torch.tensor(
        [
            [
                -(float(neg_nll[i]) - float(stem_nll[i])),
                -(float(pos_nll[i]) - float(stem_nll[i])),
            ]
            for i in range(len(labels))
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(labels, dtype=torch.long)
    return float(
        torch.nn.functional.cross_entropy(logits, target, reduction="mean").item()
    )


def accuracy_from_nll(
    stem_nll: list[float],
    neg_nll: list[float],
    pos_nll: list[float],
    labels: list[int],
) -> float:
    correct = 0
    for i, label in enumerate(labels):
        neg_suffix = float(neg_nll[i]) - float(stem_nll[i])
        pos_suffix = float(pos_nll[i]) - float(stem_nll[i])
        pred = 1 if pos_suffix < neg_suffix else 0
        correct += int(pred == int(label))
    return float(correct / len(labels)) if labels else 0.0


def hf_classification_loss(
    model,
    dataset: TorchDataset,
    collator,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    with torch.inference_mode():
        for batch in loader:
            batch = {
                k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            bsz = int(batch["labels"].numel() // int(batch["num_options"][0]))
            total += float(loss.item()) * bsz
            count += bsz
    return total / max(count, 1)


class np_random_seed:
    def __init__(self, seed: int):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc, tb):
        np.random.set_state(self.state)
