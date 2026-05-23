"""
Phase 1 Comprehensive Verification: LOZO-aligned parameters.
eps=1e-3 fixed, rank=8/16, OPT-2.7B.
"""
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
# All fp16
HF_DTYPE = torch.float16
EPS_VALUES = [1e-3]  # Only test eps=1e-3
PHASE1_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = PHASE1_DIR / "artifacts" / "adapters_phase1"

# LOZO hyperparameters
EPS = 1e-3
RANKS = [8, 16]
SEEDS_8 = list(range(42, 50))  # 8 seeds
SEEDS_4 = list(range(42, 46))  # 4 seeds

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny with clear skies.",
    "Python is a popular programming language for data science.",
    "The cat sat on the mat and watched the birds.",
    "Deep learning models require large amounts of training data.",
    "The sun rises in the east and sets in the west.",
    "Natural language processing enables computers to understand human language.",
    "Reinforcement learning agents learn from trial and error interactions.",
    "Neural networks can approximate complex nonlinear functions.",
    "Gradient descent is the foundation of modern optimization.",
    "Large language models have transformed natural language understanding.",
    "Transformers use self-attention mechanisms for sequence processing.",
    "The encoder-decoder architecture is fundamental to many NLP tasks.",
    "Word embeddings capture semantic relationships between words.",
    "Transfer learning leverages pre-trained models for new tasks.",
    "Convolutional neural networks excel at image recognition tasks.",
    "Recurrent neural networks process sequential data effectively.",
    "Batch normalization helps stabilize deep network training.",
    "Dropout is a regularization technique to prevent overfitting.",
    "Attention mechanisms allow models to focus on relevant information.",
    "Generative adversarial networks create realistic synthetic data.",
    "Autoencoders learn compressed representations of input data.",
    "Residual connections enable training of very deep networks.",
    "Knowledge distillation transfers learning from large to small models.",
    "Data augmentation increases training set diversity artificially.",
    "Learning rate schedules control the pace of model training.",
    "Weight initialization strategies affect convergence speed.",
    "Cross-entropy loss is standard for classification tasks.",
    "Mean squared error loss is used for regression problems.",
    "Adam optimizer adapts learning rates per parameter.",
    "Momentum helps accelerate gradient descent optimization.",
]

# Layer positions
LAYER_POSITIONS = {
    "early": 0,
    "middle": 15,
    "late": 31,
}

# Module names
MODULE_NAMES = ["q_proj", "v_proj", "fc1", "fc2"]


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
    """Generate LOZO-style perturbation: U, V from standard normal, no QR."""
    out_features, in_features = weight_shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    V = torch.randn(in_features, rank, generator=gen, dtype=torch.float16)
    U = torch.randn(out_features, rank, generator=gen, dtype=torch.float16)
    delta_W = eps * (U @ V.T)
    return U, V, delta_W


def save_adapter(module_path, U, V, eps, rank, sign, adapter_path):
    adapter_path = Path(adapter_path)
    if adapter_path.exists():
        shutil.rmtree(adapter_path)
    adapter_path.mkdir(parents=True)

    sign_factor = 1.0 if sign == "+" else -1.0
    lora_A = V.T.half()
    lora_B = (sign_factor * eps * U).half()

    state_dict = {
        f"base_model.model.{module_path}.lora_A.weight": lora_A,
        f"base_model.model.{module_path}.lora_B.weight": lora_B,
    }
    torch.save(state_dict, adapter_path / "adapter_model.bin")

    target_module = module_path.split(".")[-1]
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
        "target_modules": [target_module],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_adapter_multi(modules_U_V_eps, adapter_path, rank, sign="+"):
    """Save adapter with weights for multiple target modules."""
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


def get_vllm_nlls(outputs, all_ids):
    nlls = []
    for output, ids in zip(outputs, all_ids):
        prompt_lp = output.prompt_logprobs
        lps = []
        for i, lp in enumerate(prompt_lp):
            if lp is None:
                continue
            actual_id = ids[i].item()
            if isinstance(lp, dict) and actual_id in lp:
                val = lp[actual_id]
                lps.append(val.logprob if hasattr(val, "logprob") else float(val))
        nlls.append(-sum(lps) / len(lps) if lps else 0.0)
    return nlls


def run_vllm_experiment(adapter_pairs, all_ids, rank):
    """Run vLLM experiment with given adapter pairs."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=rank,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=1,
        seed=42,
    )
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=1,
    )
    prompt_token_ids = [ids.tolist() for ids in all_ids]

    # Base
    base_outputs = llm.generate(prompt_token_ids, sampling_params)
    base_nlls = get_vllm_nlls(base_outputs, all_ids)

    results = {}
    for key, (plus_path, minus_path) in adapter_pairs.items():
        plus_out = llm.generate(
            prompt_token_ids, sampling_params,
            lora_request=LoRARequest("p", 1, plus_path),
        )
        plus_nlls = get_vllm_nlls(plus_out, all_ids)

        minus_out = llm.generate(
            prompt_token_ids, sampling_params,
            lora_request=LoRARequest("m", 2, minus_path),
        )
        minus_nlls = get_vllm_nlls(minus_out, all_ids)

        deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]
        results[key] = {
            "base_nlls": base_nlls,
            "plus_nlls": plus_nlls,
            "minus_nlls": minus_nlls,
            "deltas": deltas,
        }

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: Single-layer alignment
# ─────────────────────────────────────────────────────────────────────────────
def experiment_A(model, tokenizer, seeds, tag="A"):
    """Single-layer alignment test."""
    print(f"\n{'='*70}")
    print(f"Experiment {tag}: Single-layer alignment")
    print(f"  Layers: {list(LAYER_POSITIONS.keys())}")
    print(f"  Modules: {MODULE_NAMES}")
    print(f"  Ranks: {RANKS}")
    print(f"  Seeds: {len(seeds)}")
    print(f"{'='*70}")

    batch_size = 8
    all_ids = tokenize_per_sample(tokenizer, SAMPLE_TEXTS, batch_size)
    results = []

    for layer_name, layer_idx in LAYER_POSITIONS.items():
        for module_name in MODULE_NAMES:
            module_path = get_module_path(layer_idx, module_name)
            layer = get_target_layer(model, module_path)
            W_shape = layer.weight.shape

            for seed in seeds:
                for rank in RANKS:
                    # Generate perturbation
                    U, V, delta_W = generate_lozo_perturbation(W_shape, rank, EPS, seed)
                    delta_W_device = delta_W.to(DEVICE)

                    # LOZO: W + delta
                    original = layer.weight.data.clone()
                    layer.weight.data = original + delta_W_device
                    plus_nlls = [compute_nll_single(model, ids) for ids in all_ids]

                    # LOZO: W - delta
                    layer.weight.data = original - delta_W_device
                    minus_nlls = [compute_nll_single(model, ids) for ids in all_ids]

                    # Restore
                    layer.weight.data = original

                    deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]
                    c_vals = [d / (2 * EPS) for d in deltas]

                    # Save adapters
                    plus_path = ADAPTER_DIR / f"{tag}_{layer_name}_{layer_idx}_{module_name}_r{rank}_s{seed}_plus"
                    minus_path = ADAPTER_DIR / f"{tag}_{layer_name}_{layer_idx}_{module_name}_r{rank}_s{seed}_minus"
                    save_adapter(module_path, U, V, EPS, rank, "+", plus_path)
                    save_adapter(module_path, U, V, EPS, rank, "-", minus_path)

                    results.append({
                        "tag": tag,
                        "layer_name": layer_name,
                        "layer_idx": layer_idx,
                        "module": module_name,
                        "rank": rank,
                        "seed": seed,
                        "lozo_deltas": deltas,
                        "lozo_c": c_vals,
                        "plus_path": str(plus_path),
                        "minus_path": str(minus_path),
                    })

    return results, all_ids


def run_vllm_for_A(results, all_ids):
    """Run vLLM for experiment A results."""
    for rank in RANKS:
        # Gather all adapters for this rank
        adapter_pairs = {}
        for r in results:
            if r["rank"] == rank:
                key = (r["layer_name"], r["module"], r["seed"])
                adapter_pairs[key] = (r["plus_path"], r["minus_path"])

        vllm_results = run_vllm_experiment(adapter_pairs, all_ids, rank)

        # Match results
        for r in results:
            if r["rank"] == rank:
                key = (r["layer_name"], r["module"], r["seed"])
                vr = vllm_results[key]
                r["vllm_deltas"] = vr["deltas"]
                r["vllm_base_nlls"] = vr["base_nlls"]
                r["vllm_plus_nlls"] = vr["plus_nlls"]
                r["vllm_minus_nlls"] = vr["minus_nlls"]


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: Batch size sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def experiment_B(model, tokenizer, seeds, tag="B"):
    """Batch size sensitivity test."""
    print(f"\n{'='*70}")
    print(f"Experiment {tag}: Batch size sensitivity")
    print(f"{'='*70}")

    batch_sizes = [1, 8, 16, 32]
    combos = [
        (0, "q_proj", 8),
        (15, "fc1", 8),
        (31, "v_proj", 16),
        (31, "fc2", 16),
    ]

    results = []
    for layer_idx, module_name, rank in combos:
        module_path = get_module_path(layer_idx, module_name)
        layer = get_target_layer(model, module_path)
        W_shape = layer.weight.shape

        for seed in seeds:
            U, V, delta_W = generate_lozo_perturbation(W_shape, rank, EPS, seed)
            delta_W_device = delta_W.to(DEVICE)

            for bs in batch_sizes:
                all_ids = tokenize_per_sample(tokenizer, SAMPLE_TEXTS, bs)

                # LOZO
                original = layer.weight.data.clone()
                layer.weight.data = original + delta_W_device
                plus_nlls = [compute_nll_single(model, ids) for ids in all_ids]
                layer.weight.data = original - delta_W_device
                minus_nlls = [compute_nll_single(model, ids) for ids in all_ids]
                layer.weight.data = original

                deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]

                # Save adapters
                plus_path = ADAPTER_DIR / f"{tag}_{layer_idx}_{module_name}_r{rank}_s{seed}_bs{bs}_plus"
                minus_path = ADAPTER_DIR / f"{tag}_{layer_idx}_{module_name}_r{rank}_s{seed}_bs{bs}_minus"
                save_adapter(module_path, U, V, EPS, rank, "+", plus_path)
                save_adapter(module_path, U, V, EPS, rank, "-", minus_path)

                results.append({
                    "tag": tag,
                    "layer_idx": layer_idx,
                    "module": module_name,
                    "rank": rank,
                    "seed": seed,
                    "batch_size": bs,
                    "lozo_deltas": deltas,
                    "plus_path": str(plus_path),
                    "minus_path": str(minus_path),
                })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C: Repeatability
# ─────────────────────────────────────────────────────────────────────────────
def experiment_C_setup(model, tokenizer, tag="C"):
    """Setup for repeatability test."""
    print(f"\n{'='*70}")
    print(f"Experiment {tag}: Repeatability setup")
    print(f"{'='*70}")

    batch_size = 8
    combos = [
        (0, "q_proj", 8),
        (15, "fc1", 8),
        (31, "v_proj", 16),
        (31, "fc2", 16),
    ]

    results = []
    for layer_idx, module_name, rank in combos:
        module_path = get_module_path(layer_idx, module_name)
        layer = get_target_layer(model, module_path)
        W_shape = layer.weight.shape
        seed = 42

        U, V, delta_W = generate_lozo_perturbation(W_shape, rank, EPS, seed)
        delta_W_device = delta_W.to(DEVICE)

        all_ids = tokenize_per_sample(tokenizer, SAMPLE_TEXTS, batch_size)

        # LOZO
        original = layer.weight.data.clone()
        layer.weight.data = original + delta_W_device
        plus_nlls = [compute_nll_single(model, ids) for ids in all_ids]
        layer.weight.data = original - delta_W_device
        minus_nlls = [compute_nll_single(model, ids) for ids in all_ids]
        layer.weight.data = original

        deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]

        # Save adapter once
        plus_path = ADAPTER_DIR / f"{tag}_{layer_idx}_{module_name}_r{rank}_plus"
        minus_path = ADAPTER_DIR / f"{tag}_{layer_idx}_{module_name}_r{rank}_minus"
        save_adapter(module_path, U, V, EPS, rank, "+", plus_path)
        save_adapter(module_path, U, V, EPS, rank, "-", minus_path)

        results.append({
            "tag": tag,
            "layer_idx": layer_idx,
            "module": module_name,
            "rank": rank,
            "lozo_deltas": deltas,
            "plus_path": str(plus_path),
            "minus_path": str(minus_path),
        })

    return results, all_ids


# ─────────────────────────────────────────────────────────────────────────────
# Experiment D: Multi-layer perturbation
# ─────────────────────────────────────────────────────────────────────────────
def experiment_D(model, tokenizer, seeds, tag="D"):
    """Multi-layer perturbation alignment."""
    print(f"\n{'='*70}")
    print(f"Experiment {tag}: Multi-layer perturbation")
    print(f"{'='*70}")

    batch_size = 8
    all_ids = tokenize_per_sample(tokenizer, SAMPLE_TEXTS, batch_size)

    groups = {
        # 1-layer (from layer 8)
        "L8_qv": [(8, "q_proj"), (8, "v_proj")],
        # 2-layer (from layer 8)
        "L8_9_qv": [(8, "q_proj"), (8, "v_proj"), (9, "q_proj"), (9, "v_proj")],
        # 4-layer (from layer 8)
        "L8_9_10_11_qv": [
            (8, "q_proj"), (8, "v_proj"),
            (9, "q_proj"), (9, "v_proj"),
            (10, "q_proj"), (10, "v_proj"),
            (11, "q_proj"), (11, "v_proj"),
        ],
        # 8-layer (from layer 8)
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

    results = []
    for group_name, modules in groups.items():
        for seed in seeds:
            for rank in RANKS:
                # Generate perturbations for all modules
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

                deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]

                # Save multi-module adapters
                modules_data = {}
                for module_path, pdata in perturbations.items():
                    modules_data[module_path] = (pdata["U"], pdata["V"], EPS)

                plus_path = ADAPTER_DIR / f"{tag}_{group_name}_r{rank}_s{seed}_plus"
                minus_path = ADAPTER_DIR / f"{tag}_{group_name}_r{rank}_s{seed}_minus"
                save_adapter_multi(modules_data, plus_path, rank, "+")
                save_adapter_multi(modules_data, minus_path, rank, "-")

                results.append({
                    "tag": tag,
                    "group": group_name,
                    "modules": [f"L{l}_{m}" for l, m in modules],
                    "rank": rank,
                    "seed": seed,
                    "lozo_deltas": deltas,
                    "plus_path": str(plus_path),
                    "minus_path": str(minus_path),
                })

    return results, all_ids


def print_results_table(results, tag):
    """Print results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"Results: {tag}")
    print(f"{'='*70}")

    if tag in ["A"]:
        print(f"{'layer':>8} {'module':>8} {'rank':>4} {'seed':>4} | "
              f"{'sign':>5} {'mean_c_lozo':>12} {'mean_c_vllm':>12} {'mean|c_err|':>12} {'max|c_err|':>12}")
        print("-" * 80)

        for r in results:
            if "vllm_deltas" not in r:
                continue
            n = len(r["lozo_deltas"])
            sign_matches = 0
            c_rel_errors = []
            for i in range(n):
                lc = r["lozo_c"][i]
                vc = r["vllm_deltas"][i] / (2 * EPS)
                if abs(lc) > 1e-10:
                    c_rel_errors.append(abs(vc - lc) / abs(lc))
                ls = "+" if r["lozo_deltas"][i] > 1e-8 else ("-" if r["lozo_deltas"][i] < -1e-8 else "0")
                vs = "+" if r["vllm_deltas"][i] > 1e-8 else ("-" if r["vllm_deltas"][i] < -1e-8 else "0")
                if ls == vs:
                    sign_matches += 1

            mean_c_err = sum(c_rel_errors) / len(c_rel_errors) if c_rel_errors else float("nan")
            max_c_err = max(c_rel_errors) if c_rel_errors else float("nan")
            mean_c_lozo = sum(r["lozo_c"]) / len(r["lozo_c"])
            mean_c_vllm = sum(d / (2 * EPS) for d in r["vllm_deltas"]) / len(r["vllm_deltas"])

            print(f"{r['layer_name']:>8} {r['module']:>8} {r['rank']:>4} {r['seed']:>4} | "
                  f"{sign_matches:>2}/{n:<2} {mean_c_lozo:>+12.6f} {mean_c_vllm:>+12.6f} "
                  f"{mean_c_err:>11.4%} {max_c_err:>11.4%}")

    elif tag == "D":
        print(f"{'group':>20} {'rank':>4} {'seed':>4} | "
              f"{'sign':>5} {'mean_c_lozo':>12} {'mean_c_vllm':>12} {'mean|c_err|':>12}")
        print("-" * 70)

        for r in results:
            if "vllm_deltas" not in r:
                continue
            n = len(r["lozo_deltas"])
            sign_matches = 0
            c_rel_errors = []
            for i in range(n):
                lc = r["lozo_deltas"][i] / (2 * EPS)
                vc = r["vllm_deltas"][i] / (2 * EPS)
                if abs(lc) > 1e-10:
                    c_rel_errors.append(abs(vc - lc) / abs(lc))
                ls = "+" if r["lozo_deltas"][i] > 1e-8 else ("-" if r["lozo_deltas"][i] < -1e-8 else "0")
                vs = "+" if r["vllm_deltas"][i] > 1e-8 else ("-" if r["vllm_deltas"][i] < -1e-8 else "0")
                if ls == vs:
                    sign_matches += 1

            mean_c_err = sum(c_rel_errors) / len(c_rel_errors) if c_rel_errors else float("nan")
            mean_c_lozo = sum(r["lozo_deltas"]) / len(r["lozo_deltas"]) / (2 * EPS)
            mean_c_vllm = sum(r["vllm_deltas"]) / len(r["vllm_deltas"]) / (2 * EPS)

            print(f"{r['group']:>20} {r['rank']:>4} {r['seed']:>4} | "
                  f"{sign_matches:>2}/{n:<2} {mean_c_lozo:>+12.6f} {mean_c_vllm:>+12.6f} "
                  f"{mean_c_err:>11.4%}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=HF_DTYPE, device_map=DEVICE
    )
    model.eval()

    all_results = {}

    for eps in EPS_VALUES:
        print(f"\n{'#'*70}")
        print(f"# Testing eps = {eps}")
        print(f"{'#'*70}")

        # Override global EPS
        global EPS
        EPS = eps

        # ── A-full: all layers, 4 modules, rank 8, 2 seeds ──────────────
        print(f"\n{'='*70}")
        print(f"Experiment A-full (eps={eps}): All layers")
        print(f"{'='*70}")

        results_A, all_ids_A = experiment_A(model, tokenizer, SEEDS_4[:2], f"A_eps{eps}")
        run_vllm_for_A(results_A, all_ids_A)
        print_results_table(results_A, "A")

        # Save A results
        for r in results_A:
            if "vllm_deltas" in r:
                n = len(r["lozo_deltas"])
                sign_matches = sum(1 for i in range(n)
                                   if ("+" if r["lozo_deltas"][i] > 1e-8 else ("-" if r["lozo_deltas"][i] < -1e-8 else "0"))
                                   == ("+" if r["vllm_deltas"][i] > 1e-8 else ("-" if r["vllm_deltas"][i] < -1e-8 else "0")))
                key = f"A_{eps}_{r['layer_name']}_{r['module']}_r{r['rank']}_s{r['seed']}"
                all_results[key] = {
                    "eps": eps, "type": "single", "layer": r['layer_name'], "module": r['module'],
                    "rank": r['rank'], "seed": r['seed'], "sign_match": f"{sign_matches}/{n}",
                    "mean_c_lozo": sum(r["lozo_c"]) / len(r["lozo_c"]),
                    "mean_c_vllm": sum(d / (2 * eps) for d in r["vllm_deltas"]) / len(r["vllm_deltas"]),
                }

        # ── D: Multi-layer perturbation ─────────────────────────────────
        print(f"\n{'='*70}")
        print(f"Experiment D (eps={eps}): Multi-layer perturbation")
        print(f"{'='*70}")

        results_D, all_ids_D = experiment_D(model, tokenizer, SEEDS_4[:2], f"D_eps{eps}")

        # Run vLLM for D
        for rank in RANKS:
            adapter_pairs = {}
            for r in results_D:
                if r["rank"] == rank:
                    key = r["group"] + f"_s{r['seed']}"
                    adapter_pairs[key] = (r["plus_path"], r["minus_path"])

            vllm_results = run_vllm_experiment(adapter_pairs, all_ids_D, rank)

            for r in results_D:
                if r["rank"] == rank:
                    key = r["group"] + f"_s{r['seed']}"
                    r["vllm_deltas"] = vllm_results[key]["deltas"]

        print_results_table(results_D, "D")

        # Save D results
        for r in results_D:
            if "vllm_deltas" in r:
                n = len(r["lozo_deltas"])
                sign_matches = sum(1 for i in range(n)
                                   if ("+" if r["lozo_deltas"][i] > 1e-8 else ("-" if r["lozo_deltas"][i] < -1e-8 else "0"))
                                   == ("+" if r["vllm_deltas"][i] > 1e-8 else ("-" if r["vllm_deltas"][i] < -1e-8 else "0")))
                key = f"D_{eps}_{r['group']}_r{r['rank']}_s{r['seed']}"
                all_results[key] = {
                    "eps": eps, "type": "multi", "group": r['group'],
                    "rank": r['rank'], "seed": r['seed'], "sign_match": f"{sign_matches}/{n}",
                }

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    # Group by eps
    for eps in EPS_VALUES:
        eps_results = {k: v for k, v in all_results.items() if v['eps'] == eps}
        single_results = {k: v for k, v in eps_results.items() if v['type'] == 'single'}
        multi_results = {k: v for k, v in eps_results.items() if v['type'] == 'multi'}

        # Count 100% sign match
        single_perfect = sum(1 for v in single_results.values() if v['sign_match'].startswith('8/'))
        multi_perfect = sum(1 for v in multi_results.values() if v['sign_match'].startswith('8/'))

        print(f"\neps={eps}:")
        print(f"  Single-layer: {single_perfect}/{len(single_results)} 100% sign match")
        print(f"  Multi-layer:  {multi_perfect}/{len(multi_results)} 100% sign match")

    # Save all results
    with open(f"phase1_fp16_{timestamp}.json", "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    print(f"\nResults saved to phase1_fp16_{timestamp}.json")


if __name__ == "__main__":
    main()
