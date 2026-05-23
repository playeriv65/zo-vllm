"""
Milestone 1-2: Verify in-memory LoRA vs file fake-LoRA alignment
Separate runs for file-based and in-memory adapters
"""
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

import json
import shutil
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


MODEL_NAME = "facebook/opt-2.7b"
DEVICE = "cuda"
EPS = 1e-3
RANK = 16
ADAPTER_DIR = Path("adapters_milestone1")


SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
]


def get_module_path(layer_idx, module_name):
    if module_name in ["q_proj", "v_proj"]:
        return f"model.decoder.layers.{layer_idx}.self_attn.{module_name}"
    return f"model.decoder.layers.{layer_idx}.{module_name}"


def generate_lozo_perturbation(weight_shape, rank, eps, seed):
    out_features, in_features = weight_shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    V = torch.randn(in_features, rank, generator=gen, dtype=torch.float16)
    U = torch.randn(out_features, rank, generator=gen, dtype=torch.float16)
    return U.to(DEVICE), V.to(DEVICE)


def save_adapter_file(modules_U_V_eps, adapter_path, rank, sign="+"):
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
        "base_model_name_or_path": MODEL_NAME,
        "bias": "none",
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": float(rank),
        "lora_dropout": 0.0,
        "r": rank,
        "target_modules": target_module_names,
        "task_type": "CAUSAL_LM",
    }
    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


def build_in_memory_lora_tensors(modules_U_V_eps, rank, sign="+"):
    sign_factor = 1.0 if sign == "+" else -1.0
    lora_tensors = {}
    
    for mod_name, (U, V, eps) in modules_U_V_eps.items():
        lora_A = V.T.half()
        lora_B = (sign_factor * eps * U).half()
        lora_tensors[mod_name] = {
            "lora_A": lora_A.cpu(),
            "lora_B": lora_B.cpu(),
        }
    
    return lora_tensors


def compute_nll_hf(model, input_ids, device):
    input_ids = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]
    return F.cross_entropy(shift_logits, shift_labels).item()


def get_vllm_nll(outputs, all_ids):
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


def run_file_based_test(llm, sampling_params, all_ids, modules_U_V_eps, seed, results):
    save_adapter_file(modules_U_V_eps, ADAPTER_DIR / f"plus_{seed}", RANK, sign="+")
    save_adapter_file(modules_U_V_eps, ADAPTER_DIR / f"minus_{seed}", RANK, sign="-")
    
    outputs_plus = llm.generate(
        [{"prompt_token_ids": ids.tolist()} for ids in all_ids],
        sampling_params,
        lora_request=LoRARequest(f"plus_{seed}", 1, str(ADAPTER_DIR / f"plus_{seed}")),
    )
    outputs_minus = llm.generate(
        [{"prompt_token_ids": ids.tolist()} for ids in all_ids],
        sampling_params,
        lora_request=LoRARequest(f"minus_{seed}", 2, str(ADAPTER_DIR / f"minus_{seed}")),
    )
    
    L_plus = get_vllm_nll(outputs_plus, all_ids)
    L_minus = get_vllm_nll(outputs_minus, all_ids)
    c = [(lp - lm) / (2 * EPS) for lp, lm in zip(L_plus, L_minus)]
    
    for i in range(len(all_ids)):
        results.append({
            "seed": seed,
            "sample": i,
            "method": "file",
            "L_plus": L_plus[i],
            "L_minus": L_minus[i],
            "c": c[i],
        })


def run_in_memory_test(llm, sampling_params, all_ids, modules_U_V_eps, seed, results):
    plus_tensors = build_in_memory_lora_tensors(modules_U_V_eps, RANK, sign="+")
    minus_tensors = build_in_memory_lora_tensors(modules_U_V_eps, RANK, sign="-")
    
    llm.add_lora_from_tensors(1, RANK, plus_tensors)
    llm.add_lora_from_tensors(2, RANK, minus_tensors)
    
    outputs_plus = llm.generate(
        [{"prompt_token_ids": ids.tolist()} for ids in all_ids],
        sampling_params,
        lora_request=LoRARequest("plus_mem", 1),
    )
    outputs_minus = llm.generate(
        [{"prompt_token_ids": ids.tolist()} for ids in all_ids],
        sampling_params,
        lora_request=LoRARequest("minus_mem", 2),
    )
    
    L_plus = get_vllm_nll(outputs_plus, all_ids)
    L_minus = get_vllm_nll(outputs_minus, all_ids)
    c = [(lp - lm) / (2 * EPS) for lp, lm in zip(L_plus, L_minus)]
    
    for i in range(len(all_ids)):
        results.append({
            "seed": seed,
            "sample": i,
            "method": "in_memory",
            "L_plus": L_plus[i],
            "L_minus": L_minus[i],
            "c": c[i],
        })


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ADAPTER_DIR.mkdir(exist_ok=True)
    
    print("Loading HF model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map=DEVICE)
    hf_model.eval()
    
    batch_size = 2
    all_ids = [tokenizer(text, return_tensors="pt")["input_ids"][0] for text in SAMPLE_TEXTS[:batch_size]]
    
    target_layers = [8, 9]
    target_modules = ["q_proj", "v_proj"]
    
    file_results = []
    in_memory_results = []
    
    print("Running file-based test...")
    llm_file = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=16,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        seed=42,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    
    for seed in range(42, 50):
        modules_U_V_eps = {}
        
        for layer_idx in target_layers:
            for module_name in target_modules:
                module_path = get_module_path(layer_idx, module_name)
                weight = hf_model.get_submodule(module_path).weight.data
                U, V = generate_lozo_perturbation(weight.shape, RANK, EPS, seed)
                modules_U_V_eps[module_path] = (U, V, EPS)
        
        run_file_based_test(llm_file, sampling_params, all_ids, modules_U_V_eps, seed, file_results)
    
    del llm_file
    torch.cuda.empty_cache()
    
    print("Running in-memory test...")
    llm_mem = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=16,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        seed=42,
    )
    
    for seed in range(42, 50):
        modules_U_V_eps = {}
        
        for layer_idx in target_layers:
            for module_name in target_modules:
                module_path = get_module_path(layer_idx, module_name)
                weight = hf_model.get_submodule(module_path).weight.data
                U, V = generate_lozo_perturbation(weight.shape, RANK, EPS, seed)
                modules_U_V_eps[module_path] = (U, V, EPS)
        
        run_in_memory_test(llm_mem, sampling_params, all_ids, modules_U_V_eps, seed, in_memory_results)
    
    results_dir = Path("phase2_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"milestone1_2_{timestamp}.json"
    
    all_results = {
        "file": file_results,
        "in_memory": in_memory_results,
    }
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("MILESTONE 1-2 RESULTS")
    print("=" * 80)
    
    file_c = [r["c"] for r in file_results]
    mem_c = [r["c"] for r in in_memory_results]
    
    sign_matches = sum(1 for fc, mc in zip(file_c, mem_c) if (fc > 0) == (mc > 0))
    avg_abs_err = sum(abs(fc - mc) for fc, mc in zip(file_c, mem_c)) / len(file_c)
    
    print(f"Sign Match: {sign_matches}/{len(file_c)} ({sign_matches/len(file_c)*100:.1f}%)")
    print(f"Abs Error (file vs in-mem): {avg_abs_err:.4f}")
    print("=" * 80)
    
    if sign_matches == len(file_c) and avg_abs_err < 0.01:
        print("PASS: In-memory LoRA matches file-based LoRA")
    else:
        print("FAIL: In-memory LoRA does NOT match file-based LoRA")
    
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
