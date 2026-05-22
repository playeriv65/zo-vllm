import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForTokenClassification

# Insert LOZO large_models directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "third_party", "LOZO", "large_models"))

from LOZOtrainer import LowRankTrainer
from run_lozo import OurArguments


class SimpleDataset(Dataset):
    def __init__(self, prompts, tokenizer):
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


class TrajectoryTrackingTrainer(LowRankTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = []

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader, SequentialSampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def lowrank_zo_step(self, model, inputs):
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

        # Sample the random seed for sampling 
        self.zo_random_seed = np.random.randint(1000000000)

        # Base loss (before perturbation)
        loss_base = self.zo_forward(model, inputs).item()

        # First function evaluation (+eps)
        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        loss_plus = self.zo_forward(model, inputs).item()

        # Second function evaluation (-eps)
        self.lowrank_zo_perturb_parameters(scaling_factor=-2)
        loss_minus = self.zo_forward(model, inputs).item()

        self.projected_grad = (loss_plus - loss_minus) / (2 * self.args.zo_eps)

        # Reset model back to original parameters at start of step
        self.lowrank_zo_perturb_parameters(scaling_factor=1)

        self.trajectory.append({
            "step": self.step,
            "seed": int(self.zo_random_seed),
            "loss_base": float(loss_base),
            "loss_plus": float(loss_plus),
            "loss_minus": float(loss_minus),
            "c": float(self.projected_grad)
        })

        print(f"[Baseline Helper] Step {self.step}: seed={self.zo_random_seed}, base={loss_base:.6f}, plus={loss_plus:.6f}, minus={loss_minus:.6f}, c={self.projected_grad:.6f}")

        return torch.tensor(loss_plus, device=model.device)



def main():
    model_name = "facebook/opt-2.7b"
    
    # 1. Load prompts
    prompts_file = os.path.join(project_root, "results", "input_batches.json")
    with open(prompts_file, "r") as f:
        prompts = json.load(f)
        
    print(f"[Baseline Helper] Loaded {len(prompts)} prompts for training.")

    # 2. Setup training arguments
    args = OurArguments(
        output_dir="./temp_out_baseline",
        model_name=model_name,
        learning_rate=1e-7,
        zo_eps=1e-3,
        rank_r=8,
        step_interval=100,
        trainer="LOZO",
        per_device_train_batch_size=1,  # batch size 1 for exact sequential alignment
        max_steps=len(prompts),
        evaluation_strategy="no",
        save_strategy="no",
        load_float16=True,
        only_train_option=False,
        train_as_classification=False,
        remove_unused_columns=False,
    )
    
    # Set seeds before dataset sampler setup and initialization
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # 3. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    # 4. Prepare dataset and trainer
    dataset = SimpleDataset(prompts, tokenizer)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    
    trainer = TrajectoryTrackingTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    # 5. Run training
    print("[Baseline Helper] Starting training...")
    trainer.train()
    
    # 6. Save trajectory
    trajectory_file = os.path.join(project_root, "results", "baseline_trajectory.json")
    os.makedirs(os.path.dirname(trajectory_file), exist_ok=True)
    with open(trajectory_file, "w") as f:
        json.dump(trainer.trajectory, f, indent=2)
    print(f"[Baseline Helper] Trajectory saved to {trajectory_file}")


if __name__ == "__main__":
    main()
