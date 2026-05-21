# zo-vllm

Zeroth-Order optimization with vLLM inference engine.

## Structure

- `third_party/vllm` — vLLM fork (local vendored copy)
- `third_party/LOZO` — LOZO algorithm reference (submodule)
- `zo_vllm` — core system code
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
