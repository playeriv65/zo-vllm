# Phase 2 Implementation Notes

## Current Status

### Completed
- ✅ Shard_id write for qkv_proj (can write q/k/v separately)
- ✅ HF→vLLM complete parameter mapping
- ✅ LOZO direction provider with V cache
- ✅ LoRAUpdateRuntime for all layers
- ✅ GPU-resident LoRAUpdateRuntime path for plus/minus LoRA slots

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
- vLLM accepted path is `train_scope=lora_normal`; 1D params are not sampled or
  updated in `LOZO direction provider`.

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
- vLLM accepted path uses `train_scope=lora_normal`, so embeddings remain skipped.
- Full-scope HF/LOZO ablations belong outside the reusable vLLM training path.

---

## Architecture

### Data Flow
```
LOZO direction provider
    │
    │ 1. Use LoRA-compatible parameter metadata
    │ 2. Sample U, V directions (V cached for nu steps)
    │ 3. Return direction tensors to the stepper/runtime
    │
    ▼
LoRAUpdateRuntime (GPU direct slots)
    │
    │ 4. Register stable plus/minus LoRA IDs
    │ 5. Write fixed vLLM LoRA slots in place
    │
    ▼
Scoring path
    │
    │ 6. Main runner: direct-worker compact NLL scoring
    │ 7. Legacy validation: VLLMScorer over LLM.generate(prompt_logprobs=1)
    │
    ▼
Update state / WeightSync
    │
    │ 8. Compute c = (L+ - L-) / (2*eps)
    │ 9. Apply direct base update or accumulated/LoRA-bank update state
    │ 10. Handle packed q/k/v slices
    │
    ▼
vLLM Engine (GPU)
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| LOZO direction provider | `training/direction/` | Metadata-driven U/V sampling and provider variants |
| LoRAUpdateRuntime | `core/lora_runtime/` | GPU direct plus/minus LoRA slot registration and writes |
| VLLMScorer | `experiment/scoring/generate_scorer.py` | Legacy validation loss computation via vLLM `generate()` |
| WeightSync | `weight_sync.py` | Sync updated weights or apply direct vLLM in-place updates |

---

## Known Issues

### Single-process mode required

**Issue**: the Phase 2 research harness uses `LLM.apply_model()` to reach
worker-local LoRA slots and direct weight updates. Keep it in the same UniProc
mode used by the accepted validation commands; serving-time ZO has its own
worker RPC path.

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

**Serving path**: serving-time ZO uses worker RPC and `AsyncLoRASlotRegistry`
instead of the Phase 2 validation harness.

### GPU-resident LoRA path

GPU direct LoRA slots avoid the old host round trip and the per-step
LoRAModelManager reload:

```text
CUDA U/V -> CUDA LoRA A/B -> fixed plus/minus slot
         -> BaseLayerWithLoRA.set_lora(slot_index, A/B)
```

The direct updater still uses vLLM's wrapper APIs, so packed qkv expansion,
tensor-parallel slicing, and slot layout remain centralized in vLLM.

Short validation:
- Archived CPU-mock comparison showed exact seed, U/V digest, plus/minus loss,
  and `c` match for 3/3 steps before that path was removed.
- GPU direct side-by-side vs LOZO baseline: 20/20 steps accepted,
  `direction_digest_mismatch_steps=[]`, `sign_fail_steps=[]`,
  `max_loss_plus_diff=0.009428`, `max_loss_minus_diff=0.009200`,
  `max_c_diff=6.066894` under the default
  `batch_invariant=0,enforce_eager=1` training mode.

### Direct base-weight update path

`--weight-update direct` replaces the removed full-weight copy sequence that
first updated an HF-side master tensor dict and then copied every updated
weight into vLLM.

`LOZO direction provider` now keeps only parameter metadata for direction sampling, and
the update is applied in the vLLM worker:

```text
W <- W * (1 - lr * weight_decay) - lr * c * U @ V.T
```

For packed OPT q/k/v weights, the update is applied to the corresponding
`qkv_proj.weight` slice. For other Linear weights, it is applied directly to
the base layer weight. The `float32` precision mode mirrors the plain LOZO provider's
math; the faster `param` precision mode uses the vLLM parameter dtype and true
in-place `addmm_`.

Short validation:
- Fake packed-qkv unit check: exact equality with the plain LOZO update formula in
  `float32` mode.
- Clean 20-step `direct/param` side-by-side vs LOZO baseline accepted:
  `direction_digest_mismatch_steps=[]`, `sign_fail_steps=[]`,
  `max_loss_plus_diff=0.030806`, `max_loss_minus_diff=0.023877`,
  `max_c_diff=18.560467`.
- The normal speed path keeps `direction_digest` off. Digest hashing is a
  side-by-side/debug check that copies U/V from GPU to CPU.
- Scoring is now timed internally. The clean 300-step run measured
  `step_s_mean=0.0864`, `score_s_mean=0.0585`,
  `score_generate_s_mean=0.0584`, and
  `score_postprocess_s_mean=0.000064`, so the bottleneck is vLLM
  `generate(prompt_logprobs=1)`.

### Execution flags

- Training runners do not read or set `VLLM_BATCH_INVARIANT`; it is not a
  runner setting. Dedicated sample-level batch-invariance validation may still
  set the env var internally.
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
