import os
import sys
import json
import shutil
import torch
import numpy as np
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_dataset

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "LOZO", "large_models"))

from LOZOtrainer import LowRankTrainer
from run_lozo import OurArguments
from zo_vllm.core.direction_digest import digest_named_uv


class SimpleDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        self.prompts = prompts
        self.items = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            labels = input_ids.clone()
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class ConvergenceTrainer(LowRankTrainer):
    def __init__(self, eval_batch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_batch = eval_batch
        self.eval_losses = []
        self.history = []
        self.step_count = 0

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
        args = self.args
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.step = 0
            self.v = {}

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        self.zo_random_seed = np.random.randint(1000000000)
        direction_digest = self.compute_direction_digest()

        loss_base = self.zo_forward(model, inputs).item()

        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        loss_plus = self.zo_forward(model, inputs).item()

        self.lowrank_zo_perturb_parameters(scaling_factor=-2)
        loss_minus = self.zo_forward(model, inputs).item()

        self.projected_grad = (loss_plus - loss_minus) / (2 * self.args.zo_eps)

        self.lowrank_zo_perturb_parameters(scaling_factor=1)

        self.step_count += 1

        print(f"[Baseline] Step {self.step_count}: seed={self.zo_random_seed}, base={loss_base:.6f}, plus={loss_plus:.6f}, minus={loss_minus:.6f}, c={self.projected_grad:.6f}")
        self.history.append({
            "step": self.step_count,
            "seed": int(self.zo_random_seed),
            "loss_base": float(loss_base),
            "loss_plus": float(loss_plus),
            "loss_minus": float(loss_minus),
            "c": float(self.projected_grad),
            "direction_digest": direction_digest,
            "step_s": float(time.perf_counter() - step_t0),
        })

        return torch.tensor(loss_plus, device=model.device)

    def lowrank_zo_update(self):
        super().lowrank_zo_update()
        eval_interval = getattr(self.args, "eval_interval", 20)
        if self.step_count % eval_interval == 0:
            eval_loss = self.zo_forward(self.model, self.eval_batch).item()
            self.eval_losses.append({"step": self.step_count, "loss": float(eval_loss)})
            print(f"[Baseline] Step {self.step_count} Eval Loss: {eval_loss:.6f}")


def mask_padding_labels(inputs, tokenizer):
    labels = inputs["input_ids"].clone()
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100
    elif tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. If omitted, inherit the environment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--train-scope", choices=["lora_only", "full"], default="lora_only")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    model_name = "facebook/opt-2.7b"
    rank_r = args.rank
    lr = args.lr
    zo_eps = args.eps
    step_interval = args.step_interval
    batch_size = args.batch_size
    num_steps = args.steps

    # Same seed as vLLM version: np.random.choice consumes numpy state
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load SST2 data (same np.random.choice logic as vLLM version)
    dataset_raw = load_dataset("glue", "sst2", split="train")
    num_samples = 1000
    if num_samples < len(dataset_raw):
        indices = np.random.choice(len(dataset_raw), num_samples, replace=False)
        dataset_raw = dataset_raw.select(indices)

    prompts = [f"{item['sentence']} It was" for item in dataset_raw]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    dataset = SimpleDataset(prompts, tokenizer)

    # Eval batch: first batch_size items (fixed, same as vLLM version)
    eval_batch_prompts = prompts[:batch_size]
    eval_batch = mask_padding_labels(
        dict(tokenizer(eval_batch_prompts, return_tensors="pt", padding=True)),
        tokenizer,
    )

    # Initial eval loss
    with torch.inference_mode():
        initial_loss = model(**{k: v.to(model.device) for k, v in eval_batch.items()}).loss.item()
    print(f"[Baseline] Initial loss: {initial_loss:.6f}")

    # Set numpy seed again (same as vLLM version before training loop)
    np.random.seed(args.seed)

    # max_steps = num_steps (not len(prompts))
    total_train_batch_size = batch_size * 1  # no gradient accumulation
    max_epochs = num_steps // (len(dataset) // batch_size) + 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = args.output_dir or os.path.join(
        project_root,
        "phase2",
        "results",
        "convergence",
    )
    trainer_output_dir = os.path.join(
        results_dir,
        "_tmp",
        f"baseline_trainer_{timestamp}",
    )

    our_args = OurArguments(
        output_dir=trainer_output_dir,
        model_name=model_name,
        learning_rate=lr,
        zo_eps=zo_eps,
        rank_r=rank_r,
        lozo_random_device=args.zo_random_device,
        lozo_train_scope=args.train_scope,
        step_interval=step_interval,
        trainer="LOZO",
        per_device_train_batch_size=batch_size,
        max_steps=num_steps,
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

    collator = DataCollatorForTokenClassification(tokenizer)

    trainer = ConvergenceTrainer(
        eval_batch=eval_batch,
        model=model,
        args=our_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.eval_losses.append({"step": 0, "loss": float(initial_loss)})

    print(f"[Baseline] Starting training: {num_steps} steps, batch_size={batch_size}, rank={rank_r}, lr={lr}, eps={zo_eps}")
    train_t0 = time.perf_counter()
    trainer.train()
    total_s = time.perf_counter() - train_t0

    # Final eval
    with torch.inference_mode():
        final_loss = model(**{k: v.to(model.device) for k, v in eval_batch.items()}).loss.item()
    if not trainer.eval_losses or trainer.eval_losses[-1]["step"] != num_steps:
        trainer.eval_losses.append({"step": num_steps, "loss": float(final_loss)})
    print(f"[Baseline] Final loss: {final_loss:.6f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"baseline_convergence_r{rank_r}_{timestamp}.json")
    step_times = [item["step_s"] for item in trainer.history]
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "model": model_name,
                "rank_r": rank_r,
                "lr": lr,
                "zo_eps": zo_eps,
                "step_interval": step_interval,
                "batch_size": batch_size,
                "num_steps": num_steps,
                "backend": "baseline",
                "seed": args.seed,
                "zo_random_device": args.zo_random_device,
                "train_scope": args.train_scope,
            },
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "loss_change": float(final_loss - initial_loss),
            "eval_losses": trainer.eval_losses,
            "history": trainer.history,
            "timing": {
                "total_s": float(total_s),
                "step_s_mean": float(np.mean(step_times)) if step_times else 0.0,
                "score_s_mean": 0.0,
                "sync_s_mean": 0.0,
                "lora_update_s_mean": 0.0,
            },
        }, f, indent=2)
    shutil.rmtree(trainer_output_dir, ignore_errors=True)
    try:
        os.rmdir(os.path.dirname(trainer_output_dir))
    except OSError:
        pass
    print(f"[Baseline] Results saved to {output_file}")


if __name__ == "__main__":
    main()
