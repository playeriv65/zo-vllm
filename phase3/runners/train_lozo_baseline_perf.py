import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".cache", "hf")
os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_ROOT, "datasets"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_ROOT, "hub"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(CACHE_ROOT, "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "LOZO", "large_models"))

from LOZOtrainer import LowRankTrainer  # noqa: E402
from run_lozo import OurArguments  # noqa: E402
from zo_vllm.core.direction_digest import digest_named_uv  # noqa: E402


class SimpleDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        self.items = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            self.items.append({"input_ids": input_ids, "labels": input_ids.clone()})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def mask_padding_labels(inputs, tokenizer):
    labels = inputs["input_ids"].clone()
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100
    elif tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def prepare_sst2_prompts(seed, num_samples=1000):
    np.random.seed(seed)
    dataset = load_dataset("glue", "sst2", split="train")
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)
    return [f"{item['sentence']} It was" for item in dataset]


class PerfLOZOTrainer(LowRankTrainer):
    def __init__(
        self,
        eval_batch,
        profile_mode,
        progress_interval,
        warmup_steps,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_batch = eval_batch
        self.profile_mode = profile_mode
        self.progress_interval = progress_interval
        self.warmup_steps = warmup_steps
        self.eval_losses = []
        self.history = []
        self.step_times = []
        self.zo_step_times = []
        self.update_times = []
        self.detailed_times = []
        self.step_count = 0
        self.measured_train_t0 = None
        self.measured_train_t1 = None
        self._current_step_t0 = None
        self._last_zo_step_s = 0.0
        self._last_losses = None
        self._pending_history = None
        self._pending_detail = None

    def _sync_for_detail(self):
        if self.profile_mode == "detailed" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _time_detail(self, detail, key, fn):
        if detail is None:
            return fn()
        self._sync_for_detail()
        t0 = time.perf_counter()
        result = fn()
        self._sync_for_detail()
        detail[key] = float(time.perf_counter() - t0)
        return result

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_direction_digest(self):
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self._lozo_seed_rng(self.zo_random_seed)
        digest_items = []
        try:
            for name, param in self.named_parameters_to_optim:
                if self._lozo_should_skip_param(name, param):
                    continue
                if param.data.ndim >= 2:
                    if self.step % self.args.step_interval == 0:
                        v = self._lozo_randn(
                            (param.data.size(1), self.args.rank_r),
                            param.data.device,
                            param.data.dtype,
                        )
                    else:
                        v = self.v[name]
                    u = self.random_gaussian_matrix(
                        m=param.data.size(0),
                        n=self.args.rank_r,
                        device=param.data.device,
                        dtype=param.data.dtype,
                    )
                    digest_items.append((name, u, v))
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        return digest_named_uv(digest_items)

    def lowrank_zo_step(self, model, inputs):
        step_t0 = time.perf_counter()
        self._current_step_t0 = step_t0
        detail = {} if self.profile_mode == "detailed" else None
        if hasattr(self, "step"):
            self.step += 1
        else:
            self.step = 0
            self.v = {}

        self.named_parameters_to_optim = [
            (name, param) for name, param in model.named_parameters() if param.requires_grad
        ]
        self.zo_random_seed = np.random.randint(1000000000)

        loss_base = None
        direction_digest = None
        if self.profile_mode == "instrumented":
            direction_digest = self.compute_direction_digest()
            loss_base = self.zo_forward(model, inputs).item()

        self._time_detail(
            detail,
            "perturb_plus_s",
            lambda: self.lowrank_zo_perturb_parameters(scaling_factor=1),
        )
        loss_plus = self._time_detail(
            detail,
            "forward_plus_s",
            lambda: self.zo_forward(model, inputs).item(),
        )
        self._time_detail(
            detail,
            "perturb_minus_s",
            lambda: self.lowrank_zo_perturb_parameters(scaling_factor=-2),
        )
        loss_minus = self._time_detail(
            detail,
            "forward_minus_s",
            lambda: self.zo_forward(model, inputs).item(),
        )
        self.projected_grad = (loss_plus - loss_minus) / (2 * self.args.zo_eps)
        self._time_detail(
            detail,
            "perturb_reset_s",
            lambda: self.lowrank_zo_perturb_parameters(scaling_factor=1),
        )

        self.step_count += 1
        measured_step = self.step_count > self.warmup_steps
        measured_index = self.step_count - self.warmup_steps
        if measured_step and self.measured_train_t0 is None:
            self.measured_train_t0 = step_t0
        self._last_zo_step_s = time.perf_counter() - step_t0
        self._last_losses = {
            "loss_plus": float(loss_plus),
            "loss_minus": float(loss_minus),
            "c": float(self.projected_grad),
        }

        if self.profile_mode == "instrumented" and measured_step:
            self._pending_history = {
                "step": measured_index,
                "raw_step": self.step_count,
                "seed": int(self.zo_random_seed),
                "loss_base": float(loss_base),
                "loss_plus": float(loss_plus),
                "loss_minus": float(loss_minus),
                "c": float(self.projected_grad),
                "direction_digest": direction_digest,
                "zo_step_s": float(self._last_zo_step_s),
            }
        else:
            self._pending_history = None
        self._pending_detail = detail

        return torch.tensor(loss_plus, device=model.device)

    def lowrank_zo_update(self):
        update_t0 = time.perf_counter()
        if self.profile_mode == "detailed":
            detail = self._pending_detail or {}
            self._time_detail(detail, "update_seed_s", lambda: self._lozo_seed_rng(self.zo_random_seed))

            def run_update_loop():
                args = self.args
                for name, param in self.named_parameters_to_optim:
                    if self._lozo_should_skip_param(name, param):
                        continue

                    if param.data.ndim >= 2:
                        v = self.v[name]
                        u = self.random_gaussian_matrix(
                            m=param.data.size(0),
                            n=args.rank_r,
                            device=param.data.device,
                            dtype=param.data.dtype,
                        )

                        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                            param.data = param.data - self._get_learning_rate() * (
                                self.projected_grad * (u @ v.t()) + args.weight_decay * param.data
                            )
                        else:
                            param.data = param.data - self._get_learning_rate() * (
                                self.projected_grad * (u @ v.t())
                            )
                    else:
                        z = self._lozo_randn(tuple(param.data.size()), param.data.device, param.data.dtype)
                        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                            param.data = param.data - self._get_learning_rate() * (
                                self.projected_grad * z + args.weight_decay * param.data
                            )
                        else:
                            param.data = param.data - self._get_learning_rate() * self.projected_grad * z

            self._time_detail(detail, "update_loop_s", run_update_loop)
            self._time_detail(detail, "lr_scheduler_s", self.lr_scheduler.step)
            self._pending_detail = detail
        else:
            super().lowrank_zo_update()
        update_s = time.perf_counter() - update_t0
        eval_interval = getattr(self.args, "eval_interval", 0)
        measured_step = self.step_count > self.warmup_steps
        measured_index = self.step_count - self.warmup_steps
        total_step_s = (
            time.perf_counter() - self._current_step_t0
            if self._current_step_t0 is not None
            else self._last_zo_step_s + update_s
        )
        if measured_step:
            self.step_times.append(total_step_s)
            self.zo_step_times.append(self._last_zo_step_s)
            self.update_times.append(update_s)
            if self.profile_mode == "detailed" and self._pending_detail is not None:
                self._pending_detail["update_s"] = float(update_s)
                self._pending_detail["zo_step_s"] = float(self._last_zo_step_s)
                self._pending_detail["step_s"] = float(total_step_s)
                self.detailed_times.append(self._pending_detail)
            self.measured_train_t1 = time.perf_counter()
            if self._pending_history is not None:
                self._pending_history["update_s"] = float(update_s)
                self._pending_history["step_s"] = float(total_step_s)
                self.history.append(self._pending_history)
            if (
                self.progress_interval > 0
                and measured_index % self.progress_interval == 0
                and self._last_losses is not None
            ):
                print(
                    f"[LOZO] step={measured_index} "
                    f"plus={self._last_losses['loss_plus']:.6f} "
                    f"minus={self._last_losses['loss_minus']:.6f} "
                    f"c={self._last_losses['c']:.6f} "
                    f"step_s={total_step_s:.6f} "
                    f"zo_step_s={self._last_zo_step_s:.6f} "
                    f"update_s={update_s:.6f}",
                    flush=True,
                )
        if eval_interval > 0 and measured_step and measured_index % eval_interval == 0:
            eval_loss = self.zo_forward(self.model, self.eval_batch).item()
            self.eval_losses.append({"step": measured_index, "loss": float(eval_loss)})
            print(f"[LOZO] step={measured_index} eval_loss={eval_loss:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--train-scope", choices=["lora_only", "full"], default="lora_only")
    parser.add_argument("--profile-mode", choices=["minimal", "instrumented", "detailed"], default="minimal")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--model-name", default="facebook/opt-2.7b")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    model_name = args.model_name
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.num_samples < args.batch_size:
        raise SystemExit("--num-samples must be greater than or equal to --batch-size")

    prompts = prepare_sst2_prompts(seed=args.seed, num_samples=args.num_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    if args.torch_compile:
        print(
            f"[LOZO] torch_compile_forward=true mode={args.torch_compile_mode}",
            flush=True,
        )
        model.forward = torch.compile(
            model.forward,
            mode=args.torch_compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    dataset = SimpleDataset(prompts, tokenizer)
    eval_batch = mask_padding_labels(
        dict(tokenizer(prompts[: args.batch_size], return_tensors="pt", padding=True)),
        tokenizer,
    )
    with torch.inference_mode():
        initial_loss = model(**{k: v.to(model.device) for k, v in eval_batch.items()}).loss.item()
    print(f"[LOZO] initial_loss={initial_loss:.6f}", flush=True)

    np.random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "phase3", "results", timestamp, "lozo_baseline"
    )
    trainer_output_dir = os.path.join(output_dir, "_tmp", f"trainer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    batches_per_epoch = len(dataset) // args.batch_size
    if batches_per_epoch <= 0:
        raise SystemExit("no full training batches; increase --num-samples or lower --batch-size")
    total_target_steps = args.steps + args.warmup_steps
    max_epochs = total_target_steps // batches_per_epoch + 2
    our_args = OurArguments(
        output_dir=trainer_output_dir,
        model_name=model_name,
        learning_rate=args.lr,
        zo_eps=args.eps,
        rank_r=args.rank,
        lozo_random_device=args.zo_random_device,
        lozo_train_scope=args.train_scope,
        step_interval=args.step_interval,
        trainer="LOZO",
        per_device_train_batch_size=args.batch_size,
        max_steps=total_target_steps,
        num_train_epochs=max_epochs,
        evaluation_strategy="no",
        save_strategy="no",
        load_float16=True,
        only_train_option=False,
        train_as_classification=False,
        remove_unused_columns=False,
        lr_scheduler_type="constant",
        dataloader_drop_last=True,
    )
    our_args.eval_interval = args.eval_interval
    if args.no_wandb:
        our_args.report_to = []

    trainer = PerfLOZOTrainer(
        eval_batch=eval_batch,
        profile_mode=args.profile_mode,
        progress_interval=args.progress_interval,
        warmup_steps=args.warmup_steps,
        model=model,
        args=our_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.eval_losses.append({"step": 0, "loss": float(initial_loss)})

    trainer.train()
    total_s = (
        0.0
        if trainer.measured_train_t0 is None
        else trainer.measured_train_t1 - trainer.measured_train_t0
    )

    with torch.inference_mode():
        final_loss = model(**{k: v.to(model.device) for k, v in eval_batch.items()}).loss.item()
    if not trainer.eval_losses or trainer.eval_losses[-1]["step"] != args.steps:
        trainer.eval_losses.append({"step": args.steps, "loss": float(final_loss)})
    print(f"[LOZO] final_loss={final_loss:.6f}", flush=True)

    output_file = os.path.join(output_dir, f"lozo_perf_{args.profile_mode}_{timestamp}.json")
    step_times = trainer.step_times
    zo_step_times = trainer.zo_step_times
    update_times = trainer.update_times
    detail_keys = sorted({key for item in trainer.detailed_times for key in item})
    detail_summary = {
        key: {
            "mean": float(np.mean([item[key] for item in trainer.detailed_times if key in item])),
            "std": float(np.std([item[key] for item in trainer.detailed_times if key in item])),
        }
        for key in detail_keys
    }
    with open(output_file, "w") as f:
        json.dump({
            "config": vars(args) | {"model": model_name, "backend": "lozo"},
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "loss_change": float(final_loss - initial_loss),
            "eval_losses": trainer.eval_losses,
            "history": trainer.history,
            "timing": {
                "total_s": float(total_s),
                "step_s_mean": float(np.mean(step_times)) if step_times else 0.0,
                "step_s_std": float(np.std(step_times)) if step_times else 0.0,
                "step_s_min": float(np.min(step_times)) if step_times else 0.0,
                "step_s_max": float(np.max(step_times)) if step_times else 0.0,
                "zo_step_s_mean": float(np.mean(zo_step_times)) if zo_step_times else 0.0,
                "zo_step_s_std": float(np.std(zo_step_times)) if zo_step_times else 0.0,
                "update_s_mean": float(np.mean(update_times)) if update_times else 0.0,
                "update_s_std": float(np.std(update_times)) if update_times else 0.0,
            },
            "detailed_timing": detail_summary,
        }, f, indent=2)

    shutil.rmtree(trainer_output_dir, ignore_errors=True)
    try:
        os.rmdir(os.path.dirname(trainer_output_dir))
    except OSError:
        pass
    print(f"[LOZO] saved={output_file}", flush=True)


if __name__ == "__main__":
    main()
