# zo-vllm

Zeroth-Order optimization with vLLM inference engine.

## Current Phase 2 Status

Phase 2 is in an accepted state for OPT-2.7B LOZO convergence alignment and
training-speed validation. The recommended configuration is:

```text
rank=8, step_interval=50, lr=3e-7, eps=1e-3, batch_size=16
```

Latest clean 300-step direct base-weight validation reaches 100.2% of the
instrumented baseline loss drop (`5.132858 -> 4.831391` for vLLM,
`5.132812 -> 4.832031` for baseline). The same clean run measured
`0.0864 s/step` for vLLM and `0.1155 s/step` for the instrumented baseline.
Digest hashing remains enabled only for strict side-by-side alignment. See
[`phase2/README.md`](phase2/README.md) for acceptance results, commands, and
validation notes.

Recent stepwise validation also passes with CUDA-side LOZO RNG
(`--zo-random-device cuda`): 20/20 U/V direction digests match, with max
plus/minus loss diffs of `0.030806` and `0.023877` in the 2026-05-23 clean
run.

Temporary plus/minus LoRA adapters can run through the original CPU
mock-safetensors path or the newer GPU-resident path (`--lora-residency gpu`).
The default GPU path uses `--lora-injection direct`: it initializes fixed
plus/minus slots once and overwrites their LoRA tensors in place during
training, bypassing per-step LoRA manager reload/activation. The older manager
path remains available with `--lora-injection manager`.

The training loop also has a direct base-weight update path
(`--weight-update direct --weight-update-precision param`) that applies the
LOZO low-rank update inside the vLLM worker with in-place `addmm_`, avoiding
the old external-master plus full-weight sync step. In the clean 300-step
validation, scoring remains the dominant cost (`score_s_mean=0.0585`), while
direct weight update is small (`weight_update_s_mean=0.0078`). Strict
side-by-side tests still enable digest hashing explicitly.

## Structure

- `third_party/vllm` — vLLM fork (local vendored copy)
- `third_party/LOZO` — LOZO algorithm reference (submodule)
- `phase1` — Phase 1 alignment scripts and generated Phase 1 outputs
- `phase2` — Phase 2 core code, runners, validation scripts, and generated outputs

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

Run: `.venv/bin/python phase1/phase1_official.py`

## Environment

- Python 3.12 (`.venv`)
- vLLM-ZO from `third_party/vllm` (compiled)
- Install: `uv pip install --python .venv/bin/python -e third_party/vllm --no-build-isolation`
