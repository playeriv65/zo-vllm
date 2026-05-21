"""
Test vLLM batch invariance: same sample in different batch sizes.
Does the logprob of a sample change when batch size changes?
"""
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import torch
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_NAME = "facebook/opt-2.7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

sampling_params = SamplingParams(
    temperature=0.0, max_tokens=1, prompt_logprobs=1,
)

all_ids = []
for text in SAMPLE_TEXTS:
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    all_ids.append(ids)

print("=" * 70)
print("vLLM Batch Invariance Test")
print("  Does batch size affect individual sample logprobs?")
print("=" * 70)

llm = LLM(
    model=MODEL_NAME,
    enable_lora=False,
    dtype="float16",
    max_model_len=128,
    gpu_memory_utilization=0.3,
    tensor_parallel_size=1,
    seed=42,
)

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

# Batch sizes to test
batch_sizes = [1, 2, 4, 8]

results = {}  # results[batch_size][sample_idx] = nll

for bs in batch_sizes:
    print(f"\n--- Batch size = {bs} ---")
    prompt_token_ids = [ids.tolist() for ids in all_ids[:bs]]
    outputs = llm.generate(prompt_token_ids, sampling_params)

    nlls = []
    for i, (output, ids) in enumerate(zip(outputs, all_ids[:bs])):
        nll = get_nll(output, ids)
        nlls.append(nll)
        print(f"  Sample {i}: NLL = {nll:.6f}")

    results[bs] = nlls

# Compare: for each sample, check if NLL changes across batch sizes
print("\n" + "=" * 70)
print("Comparison: Same sample across different batch sizes")
print("=" * 70)

for i in range(len(SAMPLE_TEXTS)):
    nlls_across_bs = {}
    for bs in batch_sizes:
        if i < len(results[bs]):
            nlls_across_bs[bs] = results[bs][i]

    if len(nlls_across_bs) < 2:
        continue

    values = list(nlls_across_bs.values())
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = std_val / abs(mean_val) * 100 if abs(mean_val) > 1e-10 else float('inf')
    max_diff = max(values) - min(values)
    rel_range = max_diff / abs(mean_val) * 100 if abs(mean_val) > 1e-10 else float('inf')

    print(f"\nSample {i}:")
    for bs, nll in nlls_across_bs.items():
        print(f"  bs={bs}: NLL = {nll:.6f}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std:  {std_val:.6f}")
    print(f"  CV:   {cv:.6f}%")
    print(f"  Range: {max_diff:.6f} ({rel_range:.4f}%)")

# Also test: same sample called 5 times with batch_size=1
print("\n" + "=" * 70)
print("Same sample, batch_size=1, 5 repeated calls")
print("=" * 70)

text = SAMPLE_TEXTS[0]
ids = all_ids[0]
prompt_token_ids = [ids.tolist()]

nlls_repeated = []
for i in range(5):
    outputs = llm.generate(prompt_token_ids, sampling_params)
    nll = get_nll(outputs[0], ids)
    nlls_repeated.append(nll)
    print(f"  Run {i+1}: NLL = {nll:.6f}")

mean_val = np.mean(nlls_repeated)
std_val = np.std(nlls_repeated)
cv = std_val / abs(mean_val) * 100 if abs(mean_val) > 1e-10 else float('inf')
print(f"\n  Mean: {mean_val:.6f}")
print(f"  Std:  {std_val:.6f}")
print(f"  CV:   {cv:.6f}%")

# Cleanup
del llm
torch.cuda.empty_cache()
