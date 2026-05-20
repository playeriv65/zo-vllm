# zo-vllm

Zeroth-Order optimization with vLLM inference engine.

## Structure

- `third_party/vllm` — vLLM fork (local vendored copy)
- `third_party/LOZO` — LOZO algorithm reference (submodule)
- `zo_vllm` — core system code
- `scripts` — experiment entry points
- `configs` — configuration files

## Phase 1: LOZO Perturbation Alignment

Verified that vLLM LoRA adapters can faithfully express LOZO's `±εUV^T` perturbations.

**Setup**: OPT-2.7B, layer 0 (q_proj, v_proj, fc1, fc2), rank=8, 8 samples.

**Results**:
```
rho=0.5  | sign 8/8 | gradient rel err = 0.39%
rho=1.0  | sign 8/8 | gradient rel err = 3.30%
rho=0.1  | sign 8/8 | gradient rel err = 5.78%
```

Run: `.venv312/bin/python scripts/phase1_alignment.py`

## Environment

- Python 3.12 (`.venv312`)
- vLLM-ZO from `third_party/vllm` (compiled)
- Install: `uv pip install --python .venv312/bin/python -e third_party/vllm --no-build-isolation`
