# zo-vllm

Zeroth-Order optimization with vLLM inference engine.

## Current Phase 2 Status

Phase 2 is in an accepted state for OPT-2.7B LOZO convergence alignment and
training-speed validation. The recommended configuration is:

```text
rank=8, step_interval=50, lr=3e-7, eps=1e-3, batch_size=16
```

300-step vLLM training reaches 98.6% of the baseline loss drop and runs at
2.42 steps/s versus 2.03 steps/s for the baseline. See
[`phase2/README.md`](phase2/README.md) for acceptance results, commands, and
validation notes.

Recent stepwise validation also passes with CUDA-side LOZO RNG
(`--zo-random-device cuda`): 20/20 U/V direction digests match, with max
plus/minus loss diffs near 0.01. A 100-step HF baseline ablation shows full
training scope, including embeddings and 1D params, drops loss faster than the
vLLM-compatible `lora_only` scope but not by an order of magnitude.

Temporary plus/minus LoRA adapters can run through the original CPU
mock-safetensors path or the newer GPU-resident path (`--lora-residency gpu`).
The GPU path loads CUDA PEFT tensors directly into vLLM's existing LoRA manager
and has passed short CPU/GPU residency equivalence and baseline side-by-side
smoke checks.

## Structure

- `third_party/vllm` — vLLM fork (local vendored copy)
- `third_party/LOZO` — LOZO algorithm reference (submodule)
- `phase2` — LOZO controller, memory LoRA runtime, vLLM scorer, convergence CLIs
- `scripts` — experiment entry points
- `configs` — configuration files

## Phase 1: LOZO Perturbation Alignment (Initial)

Verified that vLLM LoRA adapters can faithfully express LOZO's `±εUV^T` perturbations.

**Setup**: OPT-2.7B, layer 0 (q_proj, v_proj, fc1, fc2), rank=8, 8 samples, rho-based epsilon.

**Results**:
```
rho=0.5  | sign 8/8 | gradient rel err = 0.39%
rho=1.0  | sign 8/8 | gradient rel err = 3.30%
rho=0.1  | sign 8/8 | gradient rel err = 5.78%
```

Note: This used relative perturbation (eps = rho * ||W|| / ||UV^T||), which differs from LOZO paper.

## Phase 1.5c: LOZO-Aligned Parameters

Aligned with LOZO paper: fixed eps=1e-3, rank=8, no QR normalization.

**Setup**: OPT-2.7B, layer 0 (q_proj, v_proj, fc1, fc2), rank=8, eps=1e-3, 8 samples.

**Results**:
```
sign 8/8 | mean|c_err| = 3.29% | max|c_err| = 10.63%
```

**LOZO Default Hyperparameters** (from `third_party/LOZO/large_models/lozo.sh`):
- `EPS = 1e-3` (fixed epsilon, not relative)
- `RANK = 1` (large models) / `RANK = 4` (medium models)
- `STEP_INTERVAL = 100` (V update frequency)
- `LR = 1e-7`

**Perturbation formula** (from `LOZOtrainer.py:716`):
```python
param.data = param.data + scaling_factor * (u @ v.t()) * zo_eps
```

Run: `.venv/bin/python scripts/phase1p5c_alignment.py`

## Environment

- Python 3.12 (`.venv`)
- vLLM-ZO from `third_party/vllm` (compiled)
- Install: `uv pip install --python .venv/bin/python -e third_party/vllm --no-build-isolation`
