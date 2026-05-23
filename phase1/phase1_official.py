"""
Phase 1 Official LOZO Alignment Verification
OPT-2.7B, eps=1e-3 fixed, rank=8/16

Key: Single persistent engine, batch all requests.
"""
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

os.environ["VLLM_BATCH_INVARIANT"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_NAME = "facebook/opt-2.7b"
DEVICE = "cuda"
HF_DTYPE = torch.float16
EPS = 1e-3
PHASE1_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = PHASE1_DIR / "artifacts" / "adapters_phase1"
RESULTS_DIR = PHASE1_DIR / "results" / "official"

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


def get_module_path(layer_idx, module_name):
    if module_name in ["q_proj", "v_proj"]:
        return f"model.decoder.layers.{layer_idx}.self_attn.{module_name}"
    return f"model.decoder.layers.{layer_idx}.{module_name}"


def tokenize_per_sample(tokenizer, texts, n):
    return [tokenizer(text, return_tensors="pt")["input_ids"][0] for text in texts[:n]]


def compute_nll_single(model, input_ids):
    input_ids = input_ids.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]
    return F.cross_entropy(shift_logits, shift_labels, reduction="none").mean().item()


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
        "alpha_pattern": {}, "auto_mapping": None,
        "base_model_name_or_path": MODEL_NAME, "bias": "none",
        "exclude_modules": [], "fan_in_fan_out": False,
        "inference_mode": True, "init_lora_weights": True,
        "layers_pattern": None, "layers_to_transform": None,
        "lora_alpha": float(rank), "lora_dropout": 0.0,
        "megatron_core": "megatron.core", "megatron_config": None,
        "modules_to_save": None, "r": rank, "rank_pattern": {},
        "revision": None, "target_modules": target_module_names,
        "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False,
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


def compute_raw_data(lozo_plus, lozo_minus, vllm_plus, vllm_minus, eps):
    n = len(lozo_plus)
    raw = {
        "c_lozo": [], "c_vllm": [], "sign_match": [],
        "abs_err": [], "rel_err": [],
        "delta_lozo": [], "delta_vllm": [],
        "lozo_plus": lozo_plus, "lozo_minus": lozo_minus,
        "vllm_plus": vllm_plus, "vllm_minus": vllm_minus,
    }
    for i in range(n):
        d_lozo = lozo_plus[i] - lozo_minus[i]
        d_vllm = vllm_plus[i] - vllm_minus[i]
        c_lozo = d_lozo / (2 * eps)
        c_vllm = d_vllm / (2 * eps)
        sign_l = 1 if d_lozo > 1e-8 else (-1 if d_lozo < -1e-8 else 0)
        sign_v = 1 if d_vllm > 1e-8 else (-1 if d_vllm < -1e-8 else 0)
        raw["c_lozo"].append(c_lozo)
        raw["c_vllm"].append(c_vllm)
        raw["sign_match"].append(sign_l == sign_v)
        raw["abs_err"].append(abs(c_vllm - c_lozo))
        raw["rel_err"].append(abs(c_vllm - c_lozo) / abs(c_lozo) if abs(c_lozo) > 1e-10 else float("nan"))
        raw["delta_lozo"].append(d_lozo)
        raw["delta_vllm"].append(d_vllm)
    return raw


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HF model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=HF_DTYPE, device_map=DEVICE)
    hf_model.eval()

    batch_size = 8
    all_ids = tokenize_per_sample(tokenizer, SAMPLE_TEXTS, batch_size)

    print("Creating persistent vLLM engine (single)...")
    llm = LLM(
        model=MODEL_NAME, enable_lora=True, max_lora_rank=16,
        dtype="float16", max_model_len=128, gpu_memory_utilization=0.5,
        tensor_parallel_size=1, seed=42,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    SEEDS = list(range(42, 50))
    all_results = []
    lora_counter = 0

    # ── Table A: Single-layer alignment ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("Table A: Single-layer alignment")
    print(f"{'='*70}")

    LAYER_POSITIONS = {"early": 0, "middle": 15, "late": 31}
    MODULE_NAMES = ["q_proj", "v_proj", "fc1", "fc2"]
    RANKS = [8, 16]

    # Prepare all adapters
    table_A_tests = []
    for layer_name, layer_idx in LAYER_POSITIONS.items():
        for module_name in MODULE_NAMES:
            for rank in RANKS:
                for seed in SEEDS:
                    module_path = get_module_path(layer_idx, module_name)
                    layer = get_target_layer(hf_model, module_path)
                    U, V, delta_W = generate_lozo_perturbation(layer.weight.shape, rank, EPS, seed)

                    original = layer.weight.data.clone()
                    layer.weight.data = original + delta_W.to(DEVICE)
                    lozo_plus = [compute_nll_single(hf_model, ids) for ids in all_ids]
                    layer.weight.data = original - delta_W.to(DEVICE)
                    lozo_minus = [compute_nll_single(hf_model, ids) for ids in all_ids]
                    layer.weight.data = original

                    modules_data = {module_path: (U, V, EPS)}
                    plus_path = str(ADAPTER_DIR / f"A_{layer_name}_{module_name}_r{rank}_s{seed}_plus")
                    minus_path = str(ADAPTER_DIR / f"A_{layer_name}_{module_name}_r{rank}_s{seed}_minus")
                    save_adapter_multi(modules_data, plus_path, rank, "+")
                    save_adapter_multi(modules_data, minus_path, rank, "-")

                    table_A_tests.append({
                        "table": "A", "layer_name": layer_name, "layer_idx": layer_idx,
                        "module": module_name, "rank": rank, "seed": seed,
                        "lozo_plus": lozo_plus, "lozo_minus": lozo_minus,
                        "plus_path": plus_path, "minus_path": minus_path,
                    })

    # Batch all requests - submit all at once for vLLM scheduling
    print(f"  Prepared {len(table_A_tests)} tests, submitting all to vLLM...")
    all_requests = []
    request_map = []  # track which request belongs to which test
    for test in table_A_tests:
        lora_counter += 1
        plus_ids = [ids.tolist() for ids in all_ids]
        plus_lora = LoRARequest("p", lora_counter, test["plus_path"])
        all_requests.append((plus_ids, plus_lora, "plus"))
        request_map.append(test)
        
        lora_counter += 1
        minus_ids = [ids.tolist() for ids in all_ids]
        minus_lora = LoRARequest("m", lora_counter, test["minus_path"])
        all_requests.append((minus_ids, minus_lora, "minus"))
        request_map.append(test)
    
    # Submit all at once
    print(f"  Submitting {len(all_requests)} requests to vLLM...")
    all_outputs = []
    for ids, lora_req, sign in all_requests:
        out = llm.generate(ids, sampling_params, lora_request=lora_req)
        all_outputs.append((out, sign))
    
    # Process results
    print(f"  Processing {len(all_outputs)} results...")
    test_idx = 0
    for i, (out, sign) in enumerate(all_outputs):
        test = request_map[i]
        if sign == "plus":
            vllm_plus = get_vllm_nlls(out, all_ids)
        else:
            vllm_minus = get_vllm_nlls(out, all_ids)
            
            # Now we have both plus and minus, compute result
            raw = compute_raw_data(test["lozo_plus"], test["lozo_minus"], vllm_plus, vllm_minus, EPS)
            result = {
                "table": "A", "layer_name": test["layer_name"], "layer_idx": test["layer_idx"],
                "module": test["module"], "rank": test["rank"], "seed": test["seed"],
                "mean_c_lozo": np.mean(raw["c_lozo"]), "mean_c_vllm": np.mean(raw["c_vllm"]),
                "mean_abs_err": np.nanmean(raw["abs_err"]), "mean_rel_err": np.nanmean(raw["rel_err"]),
                "sign_matches": sum(raw["sign_match"]), "total_samples": len(raw["sign_match"]),
                "raw": raw,
            }
            all_results.append(result)
            print(f"  {test['layer_name']} {test['module']} r{test['rank']} s{test['seed']}: "
                  f"sign={result['sign_matches']}/{result['total_samples']}, abs={result['mean_abs_err']:.4f}")

    # ── Table D: Multi-layer ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Table D: Multi-layer perturbation")
    print(f"{'='*70}")

    groups_D = {
        "L0_qv": [(0, "q_proj"), (0, "v_proj")],
        "L0_fc": [(0, "fc1"), (0, "fc2")],
        "L15_all": [(15, "q_proj"), (15, "v_proj"), (15, "fc1"), (15, "fc2")],
        "L0_15_31_qv": [(0, "q_proj"), (0, "v_proj"), (15, "q_proj"), (15, "v_proj"), (31, "q_proj"), (31, "v_proj")],
    }

    table_D_tests = []
    for group_name, modules in groups_D.items():
        for rank in RANKS:
            for seed in SEEDS:
                perturbations = {}
                for layer_idx, module_name in modules:
                    module_path = get_module_path(layer_idx, module_name)
                    layer = get_target_layer(hf_model, module_path)
                    U, V, delta_W = generate_lozo_perturbation(layer.weight.shape, rank, EPS, seed)
                    perturbations[module_path] = {"U": U, "V": V, "delta_W": delta_W.to(DEVICE)}

                originals = {}
                for module_path in perturbations:
                    layer = get_target_layer(hf_model, module_path)
                    originals[module_path] = layer.weight.data.clone()
                    layer.weight.data = originals[module_path] + perturbations[module_path]["delta_W"]
                lozo_plus = [compute_nll_single(hf_model, ids) for ids in all_ids]
                for module_path in perturbations:
                    layer = get_target_layer(hf_model, module_path)
                    layer.weight.data = originals[module_path] - perturbations[module_path]["delta_W"]
                lozo_minus = [compute_nll_single(hf_model, ids) for ids in all_ids]
                for module_path in perturbations:
                    layer = get_target_layer(hf_model, module_path)
                    layer.weight.data = originals[module_path]

                modules_data = {mp: (p["U"], p["V"], EPS) for mp, p in perturbations.items()}
                plus_path = str(ADAPTER_DIR / f"D_{group_name}_r{rank}_s{seed}_plus")
                minus_path = str(ADAPTER_DIR / f"D_{group_name}_r{rank}_s{seed}_minus")
                save_adapter_multi(modules_data, plus_path, rank, "+")
                save_adapter_multi(modules_data, minus_path, rank, "-")

                table_D_tests.append({
                    "table": "D", "group": group_name, "rank": rank, "seed": seed,
                    "lozo_plus": lozo_plus, "lozo_minus": lozo_minus,
                    "plus_path": plus_path, "minus_path": minus_path,
                })

    print(f"  Prepared {len(table_D_tests)} tests, submitting all to vLLM...")
    all_requests_D = []
    request_map_D = []
    for test in table_D_tests:
        lora_counter += 1
        plus_ids = [ids.tolist() for ids in all_ids]
        plus_lora = LoRARequest("p", lora_counter, test["plus_path"])
        all_requests_D.append((plus_ids, plus_lora, "plus"))
        request_map_D.append(test)
        
        lora_counter += 1
        minus_ids = [ids.tolist() for ids in all_ids]
        minus_lora = LoRARequest("m", lora_counter, test["minus_path"])
        all_requests_D.append((minus_ids, minus_lora, "minus"))
        request_map_D.append(test)
    
    print(f"  Submitting {len(all_requests_D)} requests to vLLM...")
    all_outputs_D = []
    for ids, lora_req, sign in all_requests_D:
        out = llm.generate(ids, sampling_params, lora_request=lora_req)
        all_outputs_D.append((out, sign))
    
    print(f"  Processing {len(all_outputs_D)} results...")
    for i, (out, sign) in enumerate(all_outputs_D):
        test = request_map_D[i]
        if sign == "plus":
            vllm_plus = get_vllm_nlls(out, all_ids)
        else:
            vllm_minus = get_vllm_nlls(out, all_ids)
            
            raw = compute_raw_data(test["lozo_plus"], test["lozo_minus"], vllm_plus, vllm_minus, EPS)
            result = {
                "table": "D", "group": test["group"], "rank": test["rank"], "seed": test["seed"],
                "mean_c_lozo": np.mean(raw["c_lozo"]), "mean_c_vllm": np.mean(raw["c_vllm"]),
                "mean_abs_err": np.nanmean(raw["abs_err"]), "mean_rel_err": np.nanmean(raw["rel_err"]),
                "sign_matches": sum(raw["sign_match"]), "total_samples": len(raw["sign_match"]),
                "raw": raw,
            }
            all_results.append(result)
            print(f"  {test['group']} r{test['rank']} s{test['seed']}: "
                  f"sign={result['sign_matches']}/{result['total_samples']}, abs={result['mean_abs_err']:.4f}")

    # ── Table E: Layer count scaling ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Table E: Layer count scaling")
    print(f"{'='*70}")

    groups_E = {
        "L8_qv": [(8, "q_proj"), (8, "v_proj")],
        "L8_9_qv": [(8, "q_proj"), (8, "v_proj"), (9, "q_proj"), (9, "v_proj")],
        "L8_to_11_qv": [(i, "q_proj") for i in range(8, 12)] + [(i, "v_proj") for i in range(8, 12)],
        "L8_to_15_qv": [(i, "q_proj") for i in range(8, 16)] + [(i, "v_proj") for i in range(8, 16)],
    }

    table_E_tests = []
    for group_name, modules in groups_E.items():
        for rank in RANKS:
            for seed in SEEDS:
                perturbations = {}
                for layer_idx, module_name in modules:
                    module_path = get_module_path(layer_idx, module_name)
                    layer = get_target_layer(hf_model, module_path)
                    U, V, delta_W = generate_lozo_perturbation(layer.weight.shape, rank, EPS, seed)
                    perturbations[module_path] = {"U": U, "V": V, "delta_W": delta_W.to(DEVICE)}

                originals = {}
                for module_path in perturbations:
                    layer = get_target_layer(hf_model, module_path)
                    originals[module_path] = layer.weight.data.clone()
                    layer.weight.data = originals[module_path] + perturbations[module_path]["delta_W"]
                lozo_plus = [compute_nll_single(hf_model, ids) for ids in all_ids]
                for module_path in perturbations:
                    layer = get_target_layer(hf_model, module_path)
                    layer.weight.data = originals[module_path] - perturbations[module_path]["delta_W"]
                lozo_minus = [compute_nll_single(hf_model, ids) for ids in all_ids]
                for module_path in perturbations:
                    layer = get_target_layer(hf_model, module_path)
                    layer.weight.data = originals[module_path]

                modules_data = {mp: (p["U"], p["V"], EPS) for mp, p in perturbations.items()}
                plus_path = str(ADAPTER_DIR / f"E_{group_name}_r{rank}_s{seed}_plus")
                minus_path = str(ADAPTER_DIR / f"E_{group_name}_r{rank}_s{seed}_minus")
                save_adapter_multi(modules_data, plus_path, rank, "+")
                save_adapter_multi(modules_data, minus_path, rank, "-")

                table_E_tests.append({
                    "table": "E", "group": group_name, "rank": rank, "seed": seed,
                    "lozo_plus": lozo_plus, "lozo_minus": lozo_minus,
                    "plus_path": plus_path, "minus_path": minus_path,
                })

    print(f"  Prepared {len(table_E_tests)} tests, submitting all to vLLM...")
    all_requests_E = []
    request_map_E = []
    for test in table_E_tests:
        lora_counter += 1
        plus_ids = [ids.tolist() for ids in all_ids]
        plus_lora = LoRARequest("p", lora_counter, test["plus_path"])
        all_requests_E.append((plus_ids, plus_lora, "plus"))
        request_map_E.append(test)
        
        lora_counter += 1
        minus_ids = [ids.tolist() for ids in all_ids]
        minus_lora = LoRARequest("m", lora_counter, test["minus_path"])
        all_requests_E.append((minus_ids, minus_lora, "minus"))
        request_map_E.append(test)
    
    print(f"  Submitting {len(all_requests_E)} requests to vLLM...")
    all_outputs_E = []
    for ids, lora_req, sign in all_requests_E:
        out = llm.generate(ids, sampling_params, lora_request=lora_req)
        all_outputs_E.append((out, sign))
    
    print(f"  Processing {len(all_outputs_E)} results...")
    for i, (out, sign) in enumerate(all_outputs_E):
        test = request_map_E[i]
        if sign == "plus":
            vllm_plus = get_vllm_nlls(out, all_ids)
        else:
            vllm_minus = get_vllm_nlls(out, all_ids)
            
            raw = compute_raw_data(test["lozo_plus"], test["lozo_minus"], vllm_plus, vllm_minus, EPS)
            result = {
                "table": "E", "group": test["group"], "rank": test["rank"], "seed": test["seed"],
                "mean_c_lozo": np.mean(raw["c_lozo"]), "mean_c_vllm": np.mean(raw["c_vllm"]),
                "mean_abs_err": np.nanmean(raw["abs_err"]), "mean_rel_err": np.nanmean(raw["rel_err"]),
                "sign_matches": sum(raw["sign_match"]), "total_samples": len(raw["sign_match"]),
                "raw": raw,
            }
            all_results.append(result)
            print(f"  {test['group']} r{test['rank']} s{test['seed']}: "
                  f"sign={result['sign_matches']}/{result['total_samples']}, abs={result['mean_abs_err']:.4f}")

    # ── Table C: Repeatability ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Table C: Repeatability (10 repeats)")
    print(f"{'='*70}")

    combos_C = [(0, "q_proj", 8), (15, "fc1", 8), (31, "v_proj", 16), (31, "fc2", 16)]
    seed = 42

    all_requests_C = []
    request_map_C = []
    for layer_idx, module_name, rank in combos_C:
        module_path = get_module_path(layer_idx, module_name)
        layer = get_target_layer(hf_model, module_path)
        U, V, delta_W = generate_lozo_perturbation(layer.weight.shape, rank, EPS, seed)

        original = layer.weight.data.clone()
        layer.weight.data = original + delta_W.to(DEVICE)
        lozo_plus = [compute_nll_single(hf_model, ids) for ids in all_ids]
        layer.weight.data = original - delta_W.to(DEVICE)
        lozo_minus = [compute_nll_single(hf_model, ids) for ids in all_ids]
        layer.weight.data = original

        modules_data = {module_path: (U, V, EPS)}
        plus_path = str(ADAPTER_DIR / f"C_L{layer_idx}_{module_name}_r{rank}_plus")
        minus_path = str(ADAPTER_DIR / f"C_L{layer_idx}_{module_name}_r{rank}_minus")
        save_adapter_multi(modules_data, plus_path, rank, "+")
        save_adapter_multi(modules_data, minus_path, rank, "-")

        for r in range(10):
            lora_counter += 1
            plus_ids = [ids.tolist() for ids in all_ids]
            plus_lora = LoRARequest("p", lora_counter, plus_path)
            all_requests_C.append((plus_ids, plus_lora, "plus"))
            request_map_C.append({
                "layer_idx": layer_idx, "module": module_name, "rank": rank,
                "repeat": r, "lozo_plus": lozo_plus, "lozo_minus": lozo_minus
            })
            
            lora_counter += 1
            minus_ids = [ids.tolist() for ids in all_ids]
            minus_lora = LoRARequest("m", lora_counter, minus_path)
            all_requests_C.append((minus_ids, minus_lora, "minus"))
            request_map_C.append({
                "layer_idx": layer_idx, "module": module_name, "rank": rank,
                "repeat": r, "lozo_plus": lozo_plus, "lozo_minus": lozo_minus
            })
    
    print(f"  Submitting {len(all_requests_C)} requests to vLLM...")
    all_outputs_C = []
    for ids, lora_req, sign in all_requests_C:
        out = llm.generate(ids, sampling_params, lora_request=lora_req)
        all_outputs_C.append((out, sign))
    
    print(f"  Processing {len(all_outputs_C)} results...")
    combo_results = {}
    for i, (out, sign) in enumerate(all_outputs_C):
        test = request_map_C[i]
        key = (test["layer_idx"], test["module"], test["rank"])
        if key not in combo_results:
            combo_results[key] = {"repeats": [], "lozo_plus": test["lozo_plus"], "lozo_minus": test["lozo_minus"]}
        
        if sign == "plus":
            vllm_plus = get_vllm_nlls(out, all_ids)
            combo_results[key]["vllm_plus"] = vllm_plus
        else:
            vllm_minus = get_vllm_nlls(out, all_ids)
            combo_results[key]["vllm_minus"] = vllm_minus
            
            raw = compute_raw_data(test["lozo_plus"], test["lozo_minus"], 
                                   combo_results[key]["vllm_plus"], vllm_minus, EPS)
            combo_results[key]["repeats"].append({
                "repeat": test["repeat"], "mean_c_vllm": np.mean(raw["c_vllm"]), "raw": raw
            })
    
    for (layer_idx, module_name, rank), data in combo_results.items():
        std_c = np.std([r["mean_c_vllm"] for r in data["repeats"]])
        result = {
            "table": "C", "layer_idx": layer_idx, "module": module_name, "rank": rank,
            "repeats": data["repeats"], "std_c_vllm": std_c,
            "mean_c_lozo": np.mean(data["lozo_plus"]),
        }
        all_results.append(result)
        print(f"  L{layer_idx} {module_name} r{rank}: std_c_vllm={std_c:.6f}")

    # ── Table B: Batch size sensitivity ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("Table B: Batch size sensitivity")
    print(f"{'='*70}")

    combos_B = [(0, "q_proj", 8), (15, "fc1", 8), (31, "v_proj", 16), (31, "fc2", 16)]
    batch_sizes = [1, 8, 16, 32]

    all_requests_B = []
    request_map_B = []
    for layer_idx, module_name, rank in combos_B:
        module_path = get_module_path(layer_idx, module_name)
        layer = get_target_layer(hf_model, module_path)
        U, V, delta_W = generate_lozo_perturbation(layer.weight.shape, rank, EPS, seed)

        original = layer.weight.data.clone()
        layer.weight.data = original + delta_W.to(DEVICE)
        lozo_plus_all = [compute_nll_single(hf_model, ids) for ids in all_ids]
        layer.weight.data = original - delta_W.to(DEVICE)
        lozo_minus_all = [compute_nll_single(hf_model, ids) for ids in all_ids]
        layer.weight.data = original

        modules_data = {module_path: (U, V, EPS)}
        plus_path = str(ADAPTER_DIR / f"B_L{layer_idx}_{module_name}_r{rank}_plus")
        minus_path = str(ADAPTER_DIR / f"B_L{layer_idx}_{module_name}_r{rank}_minus")
        save_adapter_multi(modules_data, plus_path, rank, "+")
        save_adapter_multi(modules_data, minus_path, rank, "-")

        for bs in batch_sizes:
            if bs > len(all_ids):
                continue
            ids_batch = all_ids[:bs]
            lozo_plus = lozo_plus_all[:bs]
            lozo_minus = lozo_minus_all[:bs]

            for r in range(4):
                lora_counter += 1
                plus_ids = [ids.tolist() for ids in ids_batch]
                plus_lora = LoRARequest("p", lora_counter, plus_path)
                all_requests_B.append((plus_ids, plus_lora, "plus"))
                request_map_B.append({
                    "layer_idx": layer_idx, "module": module_name, "rank": rank,
                    "batch_size": bs, "repeat": r, "lozo_plus": lozo_plus, "lozo_minus": lozo_minus,
                    "ids_batch": ids_batch
                })
                
                lora_counter += 1
                minus_ids = [ids.tolist() for ids in ids_batch]
                minus_lora = LoRARequest("m", lora_counter, minus_path)
                all_requests_B.append((minus_ids, minus_lora, "minus"))
                request_map_B.append({
                    "layer_idx": layer_idx, "module": module_name, "rank": rank,
                    "batch_size": bs, "repeat": r, "lozo_plus": lozo_plus, "lozo_minus": lozo_minus,
                    "ids_batch": ids_batch
                })
    
    print(f"  Submitting {len(all_requests_B)} requests to vLLM...")
    all_outputs_B = []
    for ids, lora_req, sign in all_requests_B:
        out = llm.generate(ids, sampling_params, lora_request=lora_req)
        all_outputs_B.append((out, sign))
    
    print(f"  Processing {len(all_outputs_B)} results...")
    bs_results = {}
    for i, (out, sign) in enumerate(all_outputs_B):
        test = request_map_B[i]
        key = (test["layer_idx"], test["module"], test["rank"], test["batch_size"])
        if key not in bs_results:
            bs_results[key] = {"repeats": [], "lozo_plus": test["lozo_plus"], "lozo_minus": test["lozo_minus"]}
        
        if sign == "plus":
            vllm_plus = get_vllm_nlls(out, test["ids_batch"])
            bs_results[key]["vllm_plus"] = vllm_plus
        else:
            vllm_minus = get_vllm_nlls(out, test["ids_batch"])
            bs_results[key]["vllm_minus"] = vllm_minus
            
            raw = compute_raw_data(test["lozo_plus"], test["lozo_minus"],
                                   bs_results[key]["vllm_plus"], vllm_minus, EPS)
            bs_results[key]["repeats"].append({
                "repeat": test["repeat"], "mean_c_vllm": np.mean(raw["c_vllm"]), "raw": raw
            })
    
    for (layer_idx, module_name, rank, bs), data in bs_results.items():
        std_c = np.std([r["mean_c_vllm"] for r in data["repeats"]])
        result = {
            "table": "B", "layer_idx": layer_idx, "module": module_name,
            "rank": rank, "batch_size": bs, "repeats": data["repeats"],
            "std_c_vllm": std_c, "mean_c_lozo": np.mean(data["lozo_plus"]),
        }
        all_results.append(result)
        print(f"  L{layer_idx} {module_name} r{rank} bs={bs}: std_c_vllm={std_c:.6f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Table 1: Single-layer
    A_results = [r for r in all_results if r.get("table") == "A"]
    print("\n--- Table 1: Single-layer alignment ---")
    for layer_name in ["early", "middle", "late"]:
        for module in ["q_proj", "v_proj", "fc1", "fc2"]:
            for rank in [8, 16]:
                subset = [r for r in A_results if r["layer_name"] == layer_name and r["module"] == module and r["rank"] == rank]
                if subset:
                    sign_pct = np.mean([r["sign_matches"]/r["total_samples"] for r in subset]) * 100
                    abs_err = np.nanmean([r["mean_abs_err"] for r in subset])
                    print(f"  {layer_name:>8} {module:>8} r{rank}: sign={sign_pct:.1f}%, abs_err={abs_err:.4f}")

    # Table 5: Multi-layer
    D_results = [r for r in all_results if r.get("table") == "D"]
    print("\n--- Table 5: Multi-layer ---")
    for group in ["L0_qv", "L0_fc", "L15_all", "L0_15_31_qv"]:
        for rank in [8, 16]:
            subset = [r for r in D_results if r["group"] == group and r["rank"] == rank]
            if subset:
                sign_pct = np.mean([r["sign_matches"]/r["total_samples"] for r in subset]) * 100
                abs_err = np.nanmean([r["mean_abs_err"] for r in subset])
                print(f"  {group:>20} r{rank}: sign={sign_pct:.1f}%, abs_err={abs_err:.4f}")

    # Table 6: Layer scaling
    E_results = [r for r in all_results if r.get("table") == "E"]
    print("\n--- Table 6: Layer scaling ---")
    for group in ["L8_qv", "L8_9_qv", "L8_to_11_qv", "L8_to_15_qv"]:
        for rank in [8, 16]:
            subset = [r for r in E_results if r["group"] == group and r["rank"] == rank]
            if subset:
                sign_pct = np.mean([r["sign_matches"]/r["total_samples"] for r in subset]) * 100
                abs_err = np.nanmean([r["mean_abs_err"] for r in subset])
                print(f"  {group:>20} r{rank}: sign={sign_pct:.1f}%, abs_err={abs_err:.4f}")

    # Save
    output_file = RESULTS_DIR / f"phase1_raw_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
