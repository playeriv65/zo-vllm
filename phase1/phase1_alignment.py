"""
Phase 1: LOZO perturbation vs vLLM LoRA adapter alignment.
No padding, per-sample processing, temperature=0, fixed seeds.
"""
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "facebook/opt-2.7b"
TARGET_MODULES = [
    "model.decoder.layers.0.self_attn.q_proj",
    "model.decoder.layers.0.self_attn.v_proj",
    "model.decoder.layers.0.fc1",
    "model.decoder.layers.0.fc2",
]
RANK = 8
RHO_VALUES = [1e-1, 5e-1, 1.0]
PHASE1_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = PHASE1_DIR / "artifacts" / "adapters"
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float32

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


def tokenize_per_sample(tokenizer, texts):
    """Tokenize each text individually (no padding)."""
    all_ids = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        all_ids.append(ids)
    return all_ids


def compute_nll_single(model, input_ids):
    """Compute mean NLL for a single sequence (no padding), in fp32 for precision."""
    input_ids = input_ids.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0].float()  # cast to fp32
    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]
    per_token_nll = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return per_token_nll.mean().item()


def compute_nll_batch(model, all_ids):
    """Compute per-sample NLL (individual, no padding)."""
    model.eval()
    results = []
    for ids in all_ids:
        results.append(compute_nll_single(model, ids))
    return results


def get_target_layer(model, module_name):
    parts = module_name.split(".")
    layer = model
    for p in parts:
        layer = getattr(layer, p)
    return layer


def generate_perturbation(weight_shape, rank, rho, W_norm, seed=SEED):
    out_features, in_features = weight_shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(out_features, rank, generator=gen, dtype=torch.float64)
    V = torch.randn(in_features, rank, generator=gen, dtype=torch.float64)
    U, _ = torch.linalg.qr(U)
    V, _ = torch.linalg.qr(V)
    # ||U@V^T||_F = sqrt(rank) since orthonormal columns
    delta_norm = torch.linalg.norm(U @ V.T)
    eps = rho * W_norm / delta_norm
    return U, V, eps


def save_adapter_multi(modules_U_V_eps, adapter_path, sign="+"):
    """Save a single adapter with weights for multiple target modules."""
    adapter_path = Path(adapter_path)
    if adapter_path.exists():
        shutil.rmtree(adapter_path)
    adapter_path.mkdir(parents=True)

    sign_factor = 1.0 if sign == "+" else -1.0

    state_dict = {}
    target_module_names = []
    for mod_name, (U, V, eps) in modules_U_V_eps.items():
        lora_A = V.T.float()
        lora_B = (sign_factor * eps * U).float()
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
        "lora_alpha": float(RANK),
        "lora_dropout": 0.0,
        "megatron_core": "megatron.core",
        "megatron_config": None,
        "modules_to_save": None,
        "r": RANK,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_module_names,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return adapter_path


def run_lozo_experiment(model, all_ids):
    """LOZO ground truth: in-place perturbation on HF model, multiple layers."""
    model.eval()

    # Get all target layers
    target_layers = {}
    for mod_name in TARGET_MODULES:
        target_layers[mod_name] = get_target_layer(model, mod_name)

    # Generate one shared U,V per layer
    perturbations = {}
    for mod_name, layer in target_layers.items():
        W = layer.weight.data.clone().double().cpu()
        W_norm = torch.linalg.norm(W)
        print(f"  {mod_name}: shape={W.shape} ||W||={W_norm.item():.4f}")
        perturbations[mod_name] = {"W": W, "W_norm": W_norm}

    base_nlls = compute_nll_batch(model, all_ids)
    print(f"Base NLL: {[f'{x:.4f}' for x in base_nlls]}  mean={sum(base_nlls)/len(base_nlls):.6f}")

    results = {}
    for rho in RHO_VALUES:
        # Generate perturbations for all layers
        deltas = {}
        for mod_name, pinfo in perturbations.items():
            U, V, eps = generate_perturbation(pinfo["W"].shape, RANK, rho, pinfo["W_norm"])
            delta_W = (eps * U @ V.T).to(DTYPE).to(DEVICE)
            deltas[mod_name] = {"U": U, "V": V, "eps": eps, "delta_W": delta_W}

        # Apply all W + delta
        originals = {}
        for mod_name in target_layers:
            originals[mod_name] = target_layers[mod_name].weight.data.clone()
            target_layers[mod_name].weight.data = originals[mod_name] + deltas[mod_name]["delta_W"]
        plus_nlls = compute_nll_batch(model, all_ids)

        # Apply all W - delta
        for mod_name in target_layers:
            target_layers[mod_name].weight.data = originals[mod_name] - deltas[mod_name]["delta_W"]
        minus_nlls = compute_nll_batch(model, all_ids)

        # Restore
        for mod_name in target_layers:
            target_layers[mod_name].weight.data = originals[mod_name]

        loss_deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]
        mean_delta = sum(loss_deltas) / len(loss_deltas)

        print(f"\nrho={rho}:")
        print(f"  plus NLL:  {[f'{x:.6f}' for x in plus_nlls]}")
        print(f"  minus NLL: {[f'{x:.6f}' for x in minus_nlls]}")
        print(f"  delta:     {[f'{x:.6f}' for x in loss_deltas]}  mean={mean_delta:.8f}")

        # Save combined adapters for all modules
        rho_str = f"{rho:.0e}".replace("+", "").replace("-", "m")
        plus_adapter = {}
        minus_adapter = {}
        for mod_name, d in deltas.items():
            plus_adapter[mod_name] = (d["U"], d["V"], d["eps"])
            minus_adapter[mod_name] = (d["U"], d["V"], d["eps"])

        plus_path = ADAPTER_DIR / f"multi_plus_r{RANK}_{rho_str}"
        minus_path = ADAPTER_DIR / f"multi_minus_r{RANK}_{rho_str}"
        save_adapter_multi(plus_adapter, plus_path, "+")
        save_adapter_multi(minus_adapter, minus_path, "-")

        # Use first module's epsilon for gradient computation
        first_eps = list(deltas.values())[0]["eps"]

        results[rho] = {
            "epsilon": first_eps.item(),
            "base_nlls": base_nlls,
            "plus_nlls": plus_nlls,
            "minus_nlls": minus_nlls,
            "deltas": loss_deltas,
            "plus_path": str(plus_path),
            "minus_path": str(minus_path),
        }
    return results


def run_vllm_experiment(results, all_ids, tokenizer):
    """vLLM experiment: LoRA adapter perturbation."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"\n{'='*60}")
    print("vLLM Experiment")
    print(f"{'='*60}")

    # Pass unpadded token ID lists
    prompt_token_ids = [ids.tolist() for ids in all_ids]

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=8,
        dtype="auto",
        max_model_len=128,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=1,
        seed=42,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )

    def get_vllm_nlls(outputs, all_ids):
        """Extract per-sample NLL from vLLM outputs using actual token logprobs."""
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

    # Base
    base_outputs = llm.generate(prompt_token_ids, sampling_params)
    base_nlls = get_vllm_nlls(base_outputs, all_ids)
    print(f"Base NLL: {[f'{x:.4f}' for x in base_nlls]}  mean={sum(base_nlls)/len(base_nlls):.4f}")

    for rho, res in results.items():
        print(f"\nrho={rho}:")

        plus_outputs = llm.generate(
            prompt_token_ids, sampling_params,
            lora_request=LoRARequest("lozo_plus", 1, res["plus_path"]),
        )
        plus_nlls = get_vllm_nlls(plus_outputs, all_ids)

        minus_outputs = llm.generate(
            prompt_token_ids, sampling_params,
            lora_request=LoRARequest("lozo_minus", 2, res["minus_path"]),
        )
        minus_nlls = get_vllm_nlls(minus_outputs, all_ids)

        deltas = [p - m for p, m in zip(plus_nlls, minus_nlls)]
        mean_delta = sum(deltas) / len(deltas)

        print(f"  plus NLL:  {[f'{x:.4f}' for x in plus_nlls]}  mean={sum(plus_nlls)/len(plus_nlls):.4f}")
        print(f"  minus NLL: {[f'{x:.4f}' for x in minus_nlls]}  mean={sum(minus_nlls)/len(minus_nlls):.4f}")
        print(f"  delta:     {[f'{x:.4f}' for x in deltas]}  mean={mean_delta:.6f}")

        res["vllm_base_nlls"] = base_nlls
        res["vllm_plus_nlls"] = plus_nlls
        res["vllm_minus_nlls"] = minus_nlls
        res["vllm_deltas"] = deltas

    return results


def print_final_comparison(results):
    print(f"\n{'='*120}")
    print("FINAL COMPARISON (high precision)")
    print(f"{'='*120}")

    for rho, res in results.items():
        eps = res.get("epsilon", None)
        # If epsilon not stored, skip gradient computation
        has_eps = eps is not None

        print(f"\nrho = {rho}")
        if has_eps:
            print(f"  {'i':>2} | {'LOZO_c':>14} {'vLLM_c':>14} {'c_diff':>14} {'c_rel_err':>12} | {'LOZO_d':>12} {'vLLM_d':>12} | {'sign':>4}")
            print(f"  {'-'*2}-+-{'-'*14}-{'-'*14}-{'-'*14}-{'-'*12}-+-{'-'*12}-{'-'*12}-+-{'-'*4}")
        else:
            print(f"  {'i':>2} | {'LOZO_d':>12} {'vLLM_d':>12} {'d_diff':>12} | {'sign':>4}")
            print(f"  {'-'*2}-+-{'-'*12}-{'-'*12}-{'-'*12}-+-{'-'*4}")

        sign_matches = []
        c_rel_errors = []
        for i in range(len(res["base_nlls"])):
            ld = res["deltas"][i]
            vd = res["vllm_deltas"][i]

            ls = "+" if ld > 1e-8 else ("-" if ld < -1e-8 else "0")
            vs = "+" if vd > 1e-8 else ("-" if vd < -1e-8 else "0")
            match = "OK" if ls == vs else "NO"
            sign_matches.append(ls == vs)

            if has_eps:
                lc = ld / (2 * eps)
                vc = vd / (2 * eps)
                c_diff = vc - lc
                c_rel = abs(c_diff) / abs(lc) if abs(lc) > 1e-10 else float("nan")
                c_rel_errors.append(c_rel)
                print(f"  {i:>2} | {lc:>+14.8f} {vc:>+14.8f} {c_diff:>+14.8f} {c_rel:>11.4%} | {ld:>+12.6f} {vd:>+12.6f} | {match:>4}")
            else:
                print(f"  {i:>2} | {ld:>+12.6f} {vd:>+12.6f} {vd-ld:>+12.6f} | {match:>4}")

        n_match = sum(sign_matches)
        n_total = len(sign_matches)

        if has_eps and c_rel_errors:
            valid = [r for r in c_rel_errors if r == r]  # filter nan
            mean_c_rel = sum(valid) / len(valid) if valid else float("nan")
            print(f"\n  Sign match: {n_match}/{n_total}")
            print(f"  Mean |relative error of gradient c|: {mean_c_rel:.4%}")
            print(f"  (c = delta / (2*eps), eps = {eps:.6f})")
        else:
            print(f"\n  Sign match: {n_match}/{n_total}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_ids = tokenize_per_sample(tokenizer, SAMPLE_TEXTS)

    # Print token info
    for i, ids in enumerate(all_ids):
        tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
        print(f"Sample {i}: {len(ids)} tokens - {tokens[:5]}...")

    # Step 1-2: LOZO ground truth
    print(f"\n{'='*60}")
    print("LOZO Ground Truth (HF in-place perturbation)")
    print(f"{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    results = run_lozo_experiment(model, all_ids)
    del model
    torch.cuda.empty_cache()

    # Step 5: vLLM experiment
    results = run_vllm_experiment(results, all_ids, tokenizer)

    # Step 6: Comparison
    print_final_comparison(results)


if __name__ == "__main__":
    main()
