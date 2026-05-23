"""
Persistent vLLM engine for batch testing.
Creates engine once, then runs all tests.
"""
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import json
import shutil
import gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer

MODEL_NAME = "opt-2.7b-bf16"  # Local bf16 model
DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-3
ADAPTER_DIR = Path("adapters_batch_test")

SEEDS = list(range(42, 50))  # 8 seeds
RUNS_PER_SEED = 3

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
    return f"model.decoder.layers.{layer_idx}.{module_name}"


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
        lora_A = V.T.to(torch.float16)
        lora_B = (sign_factor * eps * U).to(torch.float16)
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
        for j, lp in enumerate(prompt_lp):
            if lp is None:
                continue
            actual_id = ids[j].item()
            if isinstance(lp, dict) and actual_id in lp:
                val = lp[actual_id]
                lps.append(val.logprob if hasattr(val, "logprob") else float(val))
        nlls.append(-sum(lps) / len(lps) if lps else 0.0)
    return nlls


def main():
    print("Loading HF model for LOZO ground truth...")
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    hf_model.to(DEVICE)
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    batch_size = 8
    all_ids = [tokenizer(t, return_tensors="pt")["input_ids"][0] for t in SAMPLE_TEXTS[:batch_size]]

    print("Creating vLLM engine (once)...")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        dtype="float16",  # Use fp16 (model is trained in fp16)
        max_model_len=128,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        max_lora_rank=16,
        seed=42,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    prompt_token_ids = [ids.tolist() for ids in all_ids]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    lora_counter = 0

    # Prepare all adapter pairs first
    all_tests = []
    for group_name, modules in GROUPS.items():
        for rank in [8, 16]:
            for seed in SEEDS:
                for run_idx in range(RUNS_PER_SEED):
                    # Generate perturbations
                    perturbations = {}
                    for layer_idx, module_name in modules:
                        module_path = get_module_path(layer_idx, module_name)
                        layer = hf_model
                        for p in module_path.split("."):
                            layer = getattr(layer, p)
                        U, V, delta_W = generate_lozo_perturbation(layer.weight.shape, rank, EPS, seed)
                        perturbations[module_path] = {"U": U, "V": V, "delta_W": delta_W.to(DEVICE)}

                    # LOZO ground truth
                    originals = {}
                    for module_path in perturbations:
                        layer = hf_model
                        for p in module_path.split("."):
                            layer = getattr(layer, p)
                        originals[module_path] = layer.weight.data.clone()
                        layer.weight.data = originals[module_path] + perturbations[module_path]["delta_W"]
                    plus_nlls_hf = []
                    for ids in all_ids:
                        with torch.no_grad():
                            logits = hf_model(ids.unsqueeze(0).to(DEVICE)).logits[0]
                        shift_logits = logits[:-1]
                        shift_labels = ids[1:].to(DEVICE)
                        nll = torch.nn.functional.cross_entropy(shift_logits, shift_labels).item()
                        plus_nlls_hf.append(nll)

                    for module_path in perturbations:
                        layer = hf_model
                        for p in module_path.split("."):
                            layer = getattr(layer, p)
                        layer.weight.data = originals[module_path] - perturbations[module_path]["delta_W"]
                    minus_nlls_hf = []
                    for ids in all_ids:
                        with torch.no_grad():
                            logits = hf_model(ids.unsqueeze(0).to(DEVICE)).logits[0]
                        shift_logits = logits[:-1]
                        shift_labels = ids[1:].to(DEVICE)
                        nll = torch.nn.functional.cross_entropy(shift_logits, shift_labels).item()
                        minus_nlls_hf.append(nll)

                    for module_path in perturbations:
                        layer = hf_model
                        for p in module_path.split("."):
                            layer = getattr(layer, p)
                        layer.weight.data = originals[module_path]

                    lozo_deltas = [p - m for p, m in zip(plus_nlls_hf, minus_nlls_hf)]

                    # Save adapters
                    modules_data = {}
                    for module_path, pdata in perturbations.items():
                        modules_data[module_path] = (pdata["U"], pdata["V"], EPS)

                    plus_path = str(ADAPTER_DIR / f"{group_name}_r{rank}_s{seed}_run{run_idx}_plus")
                    minus_path = str(ADAPTER_DIR / f"{group_name}_r{rank}_s{seed}_run{run_idx}_minus")
                    save_adapter_multi(modules_data, plus_path, rank, "+")
                    save_adapter_multi(modules_data, minus_path, rank, "-")

                    lora_counter += 1
                    all_tests.append({
                        "group": group_name, "rank": rank, "seed": seed, "run": run_idx,
                        "lozo_deltas": lozo_deltas,
                        "plus_path": plus_path, "minus_path": minus_path,
                        "plus_lora_id": lora_counter,
                    })
                    lora_counter += 1
                    all_tests[-1]["minus_lora_id"] = lora_counter

    print(f"\nPrepared {len(all_tests)} tests, starting batch vLLM inference...")

    # Batch vLLM inference
    for i, test in enumerate(all_tests):
        print(f"  [{i+1}/{len(all_tests)}] {test['group']} rank={test['rank']} seed={test['seed']} run={test['run']}", end="", flush=True)

        # Plus
        plus_out = llm.generate(prompt_token_ids, sampling_params,
                                lora_request=LoRARequest("p", test["plus_lora_id"], test["plus_path"]))
        plus_nlls_vllm = get_vllm_nlls(plus_out, all_ids)

        # Minus
        minus_out = llm.generate(prompt_token_ids, sampling_params,
                                 lora_request=LoRARequest("m", test["minus_lora_id"], test["minus_path"]))
        minus_nlls_vllm = get_vllm_nlls(minus_out, all_ids)

        vllm_deltas = [p - m for p, m in zip(plus_nlls_vllm, minus_nlls_vllm)]
        lozo_deltas = test["lozo_deltas"]

        # Calculate metrics
        sign_matches = 0
        c_rel_errors = []
        c_abs_errors = []
        c_values = []
        for j in range(len(lozo_deltas)):
            ls = "+" if lozo_deltas[j] > 1e-8 else ("-" if lozo_deltas[j] < -1e-8 else "0")
            vs = "+" if vllm_deltas[j] > 1e-8 else ("-" if vllm_deltas[j] < -1e-8 else "0")
            if ls == vs:
                sign_matches += 1
            lc = lozo_deltas[j] / (2 * EPS)
            vc = vllm_deltas[j] / (2 * EPS)
            c_values.append(abs(lc))
            if abs(lc) > 1e-10:
                c_rel_errors.append(abs(vc - lc) / abs(lc))
                c_abs_errors.append(abs(vc - lc))

        mean_c_err = sum(c_rel_errors) / len(c_rel_errors) if c_rel_errors else float("nan")
        mean_abs_err = sum(c_abs_errors) / len(c_abs_errors) if c_abs_errors else float("nan")
        max_c_value = max(c_values) if c_values else 0
        print(f"  sign={sign_matches}/8, rel_err={mean_c_err*100:.1f}%, abs_err={mean_abs_err:.4f}")

        all_results.append({
            "group": test["group"], "rank": test["rank"], "seed": test["seed"], "run": test["run"],
            "sign_match": sign_matches, "mean_c_err": mean_c_err,
            "mean_abs_err": mean_abs_err, "max_c_value": max_c_value,
            "c_values": c_values, "c_rel_errors": c_rel_errors,
            "c_abs_errors": c_abs_errors,
        })

    # Save results
    output_file = f"batch_test_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    # Print summary
    print(f"\n{'='*100}")
    print("Summary Table")
    print(f"{'='*100}")
    print(f"{'Group':<25} {'Rank':>4} {'Avg Sign':>8} {'Avg RelErr%':>10} {'Avg AbsErr':>10} {'Max RelErr%':>10}")
    print("-" * 80)

    for group_name in GROUPS:
        for rank in [8, 16]:
            group_results = [r for r in all_results if r["group"] == group_name and r["rank"] == rank]
            if not group_results:
                continue
            avg_sign = sum(r["sign_match"] for r in group_results) / len(group_results)
            avg_err = sum(r["mean_c_err"] for r in group_results) / len(group_results) * 100
            avg_abs = sum(r["mean_abs_err"] for r in group_results) / len(group_results)
            max_err = max(r["mean_c_err"] for r in group_results) * 100
            print(f"{group_name:<25} {rank:>4} {avg_sign:>8.1f} {avg_err:>10.1f} {avg_abs:>10.4f} {max_err:>10.1f}")

    # Top 5 analysis
    print(f"\n{'='*100}")
    print("Top 5 Analysis")
    print(f"{'='*100}")

    # Flatten all samples
    all_samples = []
    for r in all_results:
        for i in range(len(r.get("c_values", []))):
            all_samples.append({
                "group": r["group"], "rank": r["rank"], "seed": r["seed"], "run": r["run"],
                "c_value": r["c_values"][i],
                "c_rel_err": r["c_rel_errors"][i] if i < len(r.get("c_rel_errors", [])) else float("nan"),
                "c_abs_err": r["c_abs_errors"][i] if i < len(r.get("c_abs_errors", [])) else float("nan"),
            })

    # Top 5 by gradient magnitude
    print("\nTop 5 by gradient magnitude (|c|):")
    print(f"{'Group':<25} {'Rank':>4} {'Seed':>4} {'Run':>3} {'|c|':>10} {'RelErr%':>10} {'AbsErr':>10}")
    print("-" * 80)
    top5_by_c = sorted(all_samples, key=lambda x: x["c_value"], reverse=True)[:5]
    for s in top5_by_c:
        print(f"{s['group']:<25} {s['rank']:>4} {s['seed']:>4} {s['run']:>3} {s['c_value']:>10.4f} {s['c_rel_err']*100:>10.1f} {s['c_abs_err']:>10.4f}")

    # Top 5 by relative error
    print("\nTop 5 by relative error:")
    print(f"{'Group':<25} {'Rank':>4} {'Seed':>4} {'Run':>3} {'|c|':>10} {'RelErr%':>10} {'AbsErr':>10}")
    print("-" * 80)
    top5_by_rel = sorted(all_samples, key=lambda x: x["c_rel_err"], reverse=True)[:5]
    for s in top5_by_rel:
        print(f"{s['group']:<25} {s['rank']:>4} {s['seed']:>4} {s['run']:>3} {s['c_value']:>10.4f} {s['c_rel_err']*100:>10.1f} {s['c_abs_err']:>10.4f}")

    # Top 5 by absolute error
    print("\nTop 5 by absolute error:")
    print(f"{'Group':<25} {'Rank':>4} {'Seed':>4} {'Run':>3} {'|c|':>10} {'RelErr%':>10} {'AbsErr':>10}")
    print("-" * 80)
    top5_by_abs = sorted(all_samples, key=lambda x: x["c_abs_err"], reverse=True)[:5]
    for s in top5_by_abs:
        print(f"{s['group']:<25} {s['rank']:>4} {s['seed']:>4} {s['run']:>3} {s['c_value']:>10.4f} {s['c_rel_err']*100:>10.1f} {s['c_abs_err']:>10.4f}")

    # Analysis: Are top5 relative errors also smallest absolute errors?
    print("\n\nAnalysis: Correlation between relative error and gradient magnitude")
    print("-" * 80)
    top5_rel_c_values = [s["c_value"] for s in top5_by_rel]
    top5_abs_c_values = [s["c_value"] for s in top5_by_abs]
    print(f"Top 5 relative error samples - |c| values: {[f'{v:.4f}' for v in top5_rel_c_values]}")
    print(f"Top 5 absolute error samples - |c| values: {[f'{v:.4f}' for v in top5_abs_c_values]}")
    print(f"Average |c| for top 5 relative errors: {sum(top5_rel_c_values)/len(top5_rel_c_values):.4f}")
    print(f"Average |c| for top 5 absolute errors: {sum(top5_abs_c_values)/len(top5_abs_c_values):.4f}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
