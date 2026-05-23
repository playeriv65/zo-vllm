import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import argparse
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_NAME = "facebook/opt-2.7b"

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


def get_nll(output, ids):
    prompt_lp = output.prompt_logprobs
    lps = []
    for j, lp in enumerate(prompt_lp):
        if lp is None:
            continue
        actual_id = ids[j].item()
        if isinstance(lp, dict) and actual_id in lp:
            val = lp[actual_id]
            lps.append(val.logprob if hasattr(val, "logprob") else float(val))
    return -sum(lps) / len(lps) if lps else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify sample-level vLLM prompt-logprob invariance across batch sizes."
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. If omitted, inherit the environment.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    all_ids = [
        tokenizer(text, return_tensors="pt")["input_ids"][0]
        for text in SAMPLE_TEXTS
    ]

    print("=" * 70)
    print("vLLM Batch Invariance Test")
    print("  Does batch size affect individual sample logprobs?")
    print("=" * 70)
    print(f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT')}")

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=False,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        seed=42,
    )

    results = {}
    for bs in args.batch_sizes:
        print(f"\n--- Batch size = {bs} ---")
        prompt_token_ids = [ids.tolist() for ids in all_ids[:bs]]
        outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

        nlls = []
        for i, (output, ids) in enumerate(zip(outputs, all_ids[:bs])):
            nll = get_nll(output, ids)
            nlls.append(float(nll))
            print(f"  Sample {i}: NLL = {nll:.9f}")

        results[bs] = nlls

    print("\n" + "=" * 70)
    print("Comparison: Same sample across different batch sizes")
    print("=" * 70)

    sample_summaries = []
    max_diff_all = 0.0
    for i in range(len(SAMPLE_TEXTS)):
        nlls_across_bs = {
            bs: results[bs][i]
            for bs in args.batch_sizes
            if i < len(results[bs])
        }
        if len(nlls_across_bs) < 2:
            continue

        values = list(nlls_across_bs.values())
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        max_diff = float(max(values) - min(values))
        rel_range = max_diff / abs(mean_val) * 100 if abs(mean_val) > 1e-10 else float("inf")
        max_diff_all = max(max_diff_all, max_diff)

        print(f"\nSample {i}:")
        for bs, nll in nlls_across_bs.items():
            print(f"  bs={bs}: NLL = {nll:.9f}")
        print(f"  Mean:  {mean_val:.9f}")
        print(f"  Std:   {std_val:.9f}")
        print(f"  Range: {max_diff:.9f} ({rel_range:.6f}%)")

        sample_summaries.append({
            "sample_idx": i,
            "nll_by_batch_size": {str(k): float(v) for k, v in nlls_across_bs.items()},
            "mean": mean_val,
            "std": std_val,
            "max_diff": max_diff,
            "rel_range_pct": float(rel_range),
        })

    repeated_nlls = []
    prompt_token_ids = [all_ids[0].tolist()]
    for _ in range(5):
        outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        repeated_nlls.append(float(get_nll(outputs[0], all_ids[0])))

    repeated_max_diff = float(max(repeated_nlls) - min(repeated_nlls))
    max_diff_all = max(max_diff_all, repeated_max_diff)
    passed = max_diff_all <= args.tolerance

    print("\n" + "=" * 70)
    print("Same sample, batch_size=1, 5 repeated calls")
    print("=" * 70)
    for idx, nll in enumerate(repeated_nlls, start=1):
        print(f"  Run {idx}: NLL = {nll:.9f}")
    print(f"  Range: {repeated_max_diff:.9f}")

    summary = {
        "model": MODEL_NAME,
        "batch_sizes": args.batch_sizes,
        "tolerance": args.tolerance,
        "max_diff": max_diff_all,
        "passed": passed,
        "samples": sample_summaries,
        "repeated_batch1_nlls": repeated_nlls,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON summary to {args.output_json}")

    del llm
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"Max NLL diff: {max_diff_all:.9f}")
    print("TEST PASSED" if passed else "TEST FAILED")
    print("=" * 70)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
