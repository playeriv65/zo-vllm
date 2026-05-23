"""
Batch test: 1/2/4/8 layers from layer 8, multiple seeds, multiple runs per seed.
Usage: python batch_test.py --group L8_qv
"""
import argparse
import json
import os
import shutil
import time
import gc
from pathlib import Path
from datetime import datetime

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "facebook/opt-2.7b"
DEVICE = "cuda"
HF_DTYPE = torch.float16
EPS = 1e-3
ADAPTER_DIR = Path("adapters_batch_test")

# Test configuration
RANKS = [8, 16]
SEEDS = list(range(42, 50))  # 8 seeds: 42-49
RUNS_PER_SEED = 3  # Repeat each seed 3 times

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny with clear skies.",
    "Python is a popular programming language for data science.",
    "The cat sat on the mat and watched the birds.",
    "Deep learning models require large amounts of training data.",
    "The sun rises in the east and sets in the west.",
    "Natural language processing enables computers to understand human language.",
]

# Layer groups: from layer 8, 1/2/4/8 layers
GROUPS = {
    "L8_qv": [(8, "q_proj"), (8, "v_proj")],
    "L8_9_qv": [(8, "q_proj"), (8, "v_proj"), (9, "q_proj"), (9, "v_proj")],
    "L8_9_10_11_qv": [
        (8, "q_proj"), (8, "v_proj"),
        (9, "q_proj"), (9, "v_proj"),
        (10, "q_proj"), (10, "v_proj"),
        (11, "q_proj"), (11, "v_proj"),
    ],
    "L8_to_15_qv": [
        (8, "q_proj"), (8, "v_proj"),
        (9, "q_proj"), (9, "v_proj"),
        (10, "q_proj"), (10, "v_proj"),
        (11, "q_proj"), (11, "v_proj"),
        (12, "q_proj"), (12, "v_proj"),
        (13, "q_proj"), (13, "v_proj"),
        (14, "q_proj"), (14, "v_proj"),
        (15, "q_proj"), (15, "v_proj"),
    ],
}


def get_module_path(layer_idx, module_name):
    if module_name in ["q_proj", "v_proj"]:
        return f"model.decoder.layers.{layer_idx}.self_attn.{module_name}"
    else:
        return f"model.decoder.layers.{layer_idx}.{module_name}"


def tokenize_per_sample(tokenizer, texts, n):
    all_ids = []
    for text in texts[:n]:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        all_ids.append(ids)
    return all_ids


def compute_nll_single(model, input_ids):
    input_ids = input_ids.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]
    per_token_nll = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return per_token_nll.mean().item()


def get_target_layer(model, module_path):
    parts = module_path.split(".")
    layer = model
    for p in parts:
        layer = getattr(layer, p)
    return layer


def generate_lozo_perturbation(weight_shape, rank, eps, seed):
    out_features, in_features = weight_shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    V = torch.randn(in_features, rank, generator=gen, dtype=torch.float16)
    U = torch.randn(out_features, rank, generator=gen, dtype=torch.float16)
    delta_W = eps * (U @ V.T)
    return U, V, delta_W


def save_adapter_multi(modules_U_V_eps, adapter_path, rank, sign="+"):
    adapter_path = Path(adapter_path)
    if adapter_path.exists():
        shutil.rmtree(adapter_path)
    adapter_path.mkdir(parents=True)

    sign_factor = 1.0 if sign == "+" else -1.0
    state_dict = {}
    target_module_names = []
    for mod_name, (U, V, eps) in modules_U_V_eps.items():
        lora_A = V.T.half()
        lora_B = (sign_factor * eps * U).half()
        state_dict[f"base_model.model.{mod_name}.lora_A.weight"] = lora_A
        state_dict[f"base_model.model.{mod_name}.lora_B.weight"] = lora_B
        target_module_names.append(mod_name.split(".")[-1])

    torch.save(state_dict, adapter_path / "adapter_model.bin")

    config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": MODEL_NAME,
        "bias": "none",
        "exclude_modules": [],
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": float(rank),
        "lora_dropout": 0.0,
        "megatron_core": "megatron.core",
        "megatron_config": None,
        "modules_to_save": None,
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_module_names,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


def run_vllm_experiment(adapter_pairs, all_ids, rank):
    """Run vLLM experiment with given adapter pairs."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=1,
        max_lora_rank=rank,
        seed=42,
    )

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=1,
    )

    results = {}
    prompt_token_ids = [ids.tolist() for ids in all_ids]

    for key, (plus_path, minus_path) in adapter_pairs.items():
        # Plus
        plus_outputs = llm.generate(
            prompt_token_ids, sampling_params,
            lora_request=LoRARequest("plus", 1, plus_path),
        )
        plus_nlls = []
        for output, ids in zip(plus_outputs, all_ids):
            prompt_lp = output.prompt_logprobs
            lps = []
            for j, lp in enumerate(prompt_lp):
                if lp is None:
                    continue
                actual_id = ids[j].item()
                if isinstance(lp, dict) and actual_id in lp:
                    val = lp[actual_id]
                    lps.append(val.logprob if hasattr(val, "logprob") else float(val))
            plus_nlls.append(-sum(lps) / len(lps) if lps else 0.0)

        # Minus
        minus_outputs = llm.generate(
            prompt_token_ids, sampling_params,
            lora_request=LoRARequest("minus", 2, minus_path),
        )
        minus_nlls = []
        for output, ids in zip(minus_outputs, all_ids):
            prompt_lp = output.prompt_logprobs
            lps = []
            for j, lp in enumerate(prompt_lp):
                if lp is None:
                    continue
                actual_id = ids[j].item()
                if isinstance(lp, dict) and actual_id in lp:
                    val = lp[actual_id]
                    lps.append(val.logprob if hasattr(val, "logprob") else float(val))
            minus_nlls.append(-sum(lps) / len(lps) if lps else 0.0)

        deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]
        results[key] = {"deltas": deltas}

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default=None, help="Run only this group")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Batch test started at {timestamp}")
    print(f"EPS={EPS}, RANKS={RANKS}, SEEDS={SEEDS}, RUNS_PER_SEED={RUNS_PER_SEED}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=HF_DTYPE)
    model.to(DEVICE)
    model.eval()

    batch_size = 8
    all_ids = tokenize_per_sample(tokenizer, SAMPLE_TEXTS, batch_size)

    all_results = []

    groups_to_run = {args.group: GROUPS[args.group]} if args.group else GROUPS

    for group_name, modules in groups_to_run.items():
        print(f"\n{'='*70}")
        print(f"Group: {group_name} ({len(modules)//2} layers)")
        print(f"{'='*70}")

        for rank in RANKS:
            for seed in SEEDS:
                for run_idx in range(RUNS_PER_SEED):
                    print(f"\n  rank={rank}, seed={seed}, run={run_idx+1}/{RUNS_PER_SEED}")

                    # Generate perturbations
                    perturbations = {}
                    for layer_idx, module_name in modules:
                        module_path = get_module_path(layer_idx, module_name)
                        layer = get_target_layer(model, module_path)
                        W_shape = layer.weight.shape
                        U, V, delta_W = generate_lozo_perturbation(W_shape, rank, EPS, seed)
                        perturbations[module_path] = {
                            "U": U, "V": V, "delta_W": delta_W.to(DEVICE),
                        }

                    # LOZO: W + delta
                    originals = {}
                    for module_path in perturbations:
                        layer = get_target_layer(model, module_path)
                        originals[module_path] = layer.weight.data.clone()
                        layer.weight.data = originals[module_path] + perturbations[module_path]["delta_W"]
                    plus_nlls = [compute_nll_single(model, ids) for ids in all_ids]

                    # LOZO: W - delta
                    for module_path in perturbations:
                        layer = get_target_layer(model, module_path)
                        layer.weight.data = originals[module_path] - perturbations[module_path]["delta_W"]
                    minus_nlls = [compute_nll_single(model, ids) for ids in all_ids]

                    # Restore
                    for module_path in perturbations:
                        layer = get_target_layer(model, module_path)
                        layer.weight.data = originals[module_path]

                    lozo_deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]

                    # Save adapters
                    modules_data = {}
                    for module_path, pdata in perturbations.items():
                        modules_data[module_path] = (pdata["U"], pdata["V"], EPS)

                    plus_path = ADAPTER_DIR / f"{group_name}_r{rank}_s{seed}_run{run_idx}_plus"
                    minus_path = ADAPTER_DIR / f"{group_name}_r{rank}_s{seed}_run{run_idx}_minus"
                    save_adapter_multi(modules_data, plus_path, rank, "+")
                    save_adapter_multi(modules_data, minus_path, rank, "-")

                    # Run vLLM
                    adapter_pairs = {"test": (str(plus_path), str(minus_path))}
                    vllm_results = run_vllm_experiment(adapter_pairs, all_ids, rank)
                    vllm_deltas = vllm_results["test"]["deltas"]

                    # Calculate sign match
                    sign_matches = 0
                    for i in range(len(lozo_deltas)):
                        ls = "+" if lozo_deltas[i] > 1e-8 else ("-" if lozo_deltas[i] < -1e-8 else "0")
                        vs = "+" if vllm_deltas[i] > 1e-8 else ("-" if vllm_deltas[i] < -1e-8 else "0")
                        if ls == vs:
                            sign_matches += 1

                    # Calculate relative error
                    c_rel_errors = []
                    for i in range(len(lozo_deltas)):
                        lc = lozo_deltas[i] / (2 * EPS)
                        vc = vllm_deltas[i] / (2 * EPS)
                        if abs(lc) > 1e-10:
                            c_rel_errors.append(abs(vc - lc) / abs(lc))

                    mean_c_err = sum(c_rel_errors) / len(c_rel_errors) if c_rel_errors else float("nan")

                    result = {
                        "group": group_name,
                        "num_layers": len(modules) // 2,
                        "rank": rank,
                        "seed": seed,
                        "run": run_idx,
                        "sign_match": sign_matches,
                        "mean_c_err": mean_c_err,
                        "lozo_deltas": lozo_deltas,
                        "vllm_deltas": vllm_deltas,
                    }
                    all_results.append(result)

                    print(f"    sign={sign_matches}/8, err={mean_c_err*100:.1f}%")

    # Save all results
    output_file = f"batch_test_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    # Print summary table
    print(f"\n{'='*100}")
    print("Summary Table")
    print(f"{'='*100}")
    print(f"{'Group':<25} {'Rank':>4} {'Num Seeds':>9} {'Num Runs':>8} {'Avg Sign':>8} {'Avg Err%':>10} {'Max Err%':>10} {'Min Err%':>10}")
    print("-" * 100)

    for group_name in GROUPS:
        for rank in RANKS:
            group_results = [r for r in all_results if r["group"] == group_name and r["rank"] == rank]
            if not group_results:
                continue

            avg_sign = sum(r["sign_match"] for r in group_results) / len(group_results)
            avg_err = sum(r["mean_c_err"] for r in group_results) / len(group_results) * 100
            max_err = max(r["mean_c_err"] for r in group_results) * 100
            min_err = min(r["mean_c_err"] for r in group_results) * 100

            print(f"{group_name:<25} {rank:>4} {len(SEEDS):>9} {len(group_results)//len(SEEDS):>8} {avg_sign:>8.1f} {avg_err:>10.1f} {max_err:>10.1f} {min_err:>10.1f}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
