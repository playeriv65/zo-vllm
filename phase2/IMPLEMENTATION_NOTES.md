# Phase 2 Implementation Notes

## Current Status

### Completed
- ✅ Shard_id write for qkv_proj (can write q/k/v separately)
- ✅ HF→vLLM complete parameter mapping
- ✅ LOZOController with V cache
- ✅ TempLoRARuntime for all layers
- ✅ GPU-resident TempLoRARuntime path for plus/minus LoRA slots

### Simplifications (accepted vLLM scope)

#### 1D Parameters Skipped

**Reason**: vLLM LoRA only supports 2D parameters (Linear layers).

**LOZO baseline behavior** (from `LOZOtrainer.py` line 708-719):
```python
if param.ndim >= 2:  # 2D params
    # Low-rank perturbation: u @ v.T
    v = torch.randn(in_features, rank_r)
    u = torch.randn(out_features, rank_r)
    param.data += scaling_factor * (u @ v.T) * zo_eps
else:  # 1D params (bias, layer_norm.weight/bias)
    # Full-rank perturbation: z
    z = torch.normal(mean=0, std=1, size=param.size())
    param.data += scaling_factor * z * zo_eps
```

**Affected parameters**:
- `self_attn_layer_norm.weight`: [2560]
- `self_attn_layer_norm.bias`: [2560]
- `final_layer_norm.weight`: [2560]
- `final_layer_norm.bias`: [2560]
- Linear layers' bias (if present)

**Current handling**:
- vLLM accepted path uses `train_scope=lora_only`, so 1D params remain skipped.
- HF/LOZO baseline supports `train_scope=full` for ablations that include 1D params.

---

#### Embedding Parameters Skipped

**Reason**: vLLM LoRA only supports Linear layers, not embedding layers.

**LOZO baseline behavior**: LOZO perturbs **all** trainable 2D parameters, including:
- `model.decoder.embed_tokens.weight`: [vocab_size, hidden_size]
- `model.decoder.embed_positions.weight`: [max_position, hidden_size]

**Affected parameters**:
- `embed_tokens.weight`: Token embeddings
- `embed_positions.weight`: Position embeddings (OPT uses learned position embeddings)

**Current handling**:
- vLLM accepted path uses `train_scope=lora_only`, so embeddings remain skipped.
- HF/LOZO baseline supports `train_scope=full` for ablations that include embeddings.
- 100-step baseline ablation with CUDA RNG: `full` drops eval loss by `0.250000`;
  `lora_only` drops by `0.140625` under the same hyperparameters.

---

## Architecture

### Data Flow
```
LOZOController (HF model master weights on CPU or CUDA)
    │
    │ 1. Maintain master weights (LoRA-compatible params by default)
    │ 2. Sample U, V directions (V cached for step_interval steps)
    │ 3. Build LoRA tensors for perturbation forward
    │
    ▼
TempLoRARuntime (CPU mock or GPU-resident LoRA)
    │
    │ 4. Register/select stable plus/minus LoRA IDs
    │ 5. Update LoRA tensors each step
    │    - cpu: mock safetensors path + LoRARequest(load_inplace=True)
    │    - gpu: LoRAModel.from_lora_tensors + model.lora_manager
    │
    ▼
VLLMScorer
    │
    │ 6. Compute loss_plus, loss_minus via LoRA forward
    │
    ▼
LOZOController
    │
    │ 7. Compute c = (L+ - L-) / (2*eps)
    │ 8. Update master weights
    │
    ▼
WeightSync
    │
    │ 9. Map HF names → vLLM names
    │ 10. Write q/k/v via shard_id
    │ 11. Write other params directly
    │
    ▼
vLLM Engine (GPU)
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| LOZOController | `lozo_controller.py` | Master weights, U/V sampling, updates |
| TempLoRARuntime | `temp_lora_runtime.py` | CPU mock or GPU-resident LoRA for perturbation |
| VLLMScorer | `vllm_scorer.py` | Loss computation via vLLM |
| WeightSync | `weight_sync.py` | Sync updated weights to vLLM |

---

## Known Issues

### Single-process mode required

**Issue**: CPU mock functions only work in single-process mode. The
GPU-resident path also currently uses `LLM.apply_model()` and is validated for
the same UniProc research harness, not general vLLM multi-process serving.

**Environment variables**:
```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0  # Disable multi-process
VLLM_ALLOW_INSECURE_SERIALIZATION=1  # Allow pickle serialization
```

**Performance impact**: single-process mode is required for this research
harness, but CUDA graphs are controlled independently by `--enforce-eager`.
Current default is `--enforce-eager 1` for low startup cost; set
`--enforce-eager 0` only when the run is long enough to amortize compile and
graph-capture time.

**Future work**: Add a first-class vLLM worker RPC for in-memory CUDA LoRA
tensors if multi-process serving becomes a requirement.

### GPU-resident LoRA path

`--lora-residency gpu` avoids the old host round trip:

```text
CUDA U/V -> CUDA LoRA A/B -> LoRAModel.from_lora_tensors(device=manager.device)
         -> LoRAModelManager.add_adapter/activate_adapter -> GPU LoRA slots
```

This keeps vLLM's existing packed qkv handling, tensor-parallel slicing, and
slot metadata logic. It does not write `lora_a_stacked` or `lora_b_stacked`
directly from phase2 code.

Short validation:
- vLLM CPU mock vs GPU residency: exact seed, U/V digest, plus/minus loss, and
  `c` match for 3/3 steps.
- GPU-resident side-by-side vs LOZO baseline: 3/3 steps accepted,
  `direction_digest_mismatch_steps=[]`, `max_loss_plus_diff=0.007089`,
  `max_loss_minus_diff=0.001356`, `max_c_diff=2.866773` under the default
  `batch_invariant=0,enforce_eager=1` training mode.

### Execution flags

- `--batch-invariant 0` is the training default. Use `--batch-invariant 1` only
  for explicit sample-level batch-invariance validation or reproducing older
  accepted runs.
- `--enforce-eager 1` is the default because it keeps vLLM engine startup low.
  A 20-step GPU-resident timing ablation found the fastest loop at
  `batch_invariant=0,enforce_eager=0` (`step_s_mean=0.2092`,
  `tail10_step_s_mean=0.2001`), but cached vLLM initialization still took
  `23.05s` on the `batch_invariant=1,enforce_eager=0` run because of
  torch.compile and CUDA graph capture.
- All four `batch_invariant`/`enforce_eager` combinations produced identical
  step seeds and U/V direction digests in the 20-step ablation.

---

## RNG And Ablation Notes

- `--seed` controls the numpy stream that produces the per-step ZO seeds.
- `--zo-random-device cpu|cuda` controls where U/V/z tensors are sampled.
- Phase 2 CLIs default to CUDA RNG for speed. CPU RNG and CUDA RNG are not
  bitwise-identical streams; CPU RNG remains available only to reproduce older
  CPU-RNG experiments.
- CUDA RNG side-by-side is accepted: 20/20 U/V digests match and all
  plus/minus loss and `c` differences pass tolerance.
- Embedding/1D support remains a baseline-only ablation unless vLLM gains
  a direct non-LoRA perturbation path for those tensors.
