# Phase 2 Implementation Notes

## Current Status

### Completed
- ✅ Shard_id write for qkv_proj (can write q/k/v separately)
- ✅ HF→vLLM complete parameter mapping
- ✅ LOZOController with V cache
- ✅ TempLoRARuntime for all layers

### Simplifications (for initial implementation)

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

**Future work**: 
- Option A: Directly modify base weights for 1D params (bypass LoRA)
- Option B: Accept simplification and compare with baseline

---

#### Embedding Parameters Skipped

**Reason**: vLLM LoRA only supports Linear layers, not embedding layers.

**LOZO baseline behavior**: LOZO perturbs **all** trainable 2D parameters, including:
- `model.decoder.embed_tokens.weight`: [vocab_size, hidden_size]
- `model.decoder.embed_positions.weight`: [max_position, hidden_size]

**Affected parameters**:
- `embed_tokens.weight`: Token embeddings
- `embed_positions.weight`: Position embeddings (OPT uses learned position embeddings)

**Future work**: 
- Modify vLLM source code to support embedding perturbation via:
  - Option A: Extend LoRA to support embedding layers
  - Option B: Temporarily modify base weights before/after forward pass
  - Option C: Use hooks to inject embedding perturbations

---

## Architecture

### Data Flow
```
LOZOController (HF model on CPU)
    │
    │ 1. Maintain master weights (all 2D trainable params)
    │ 2. Sample U, V directions (V cached for step_interval steps)
    │ 3. Build LoRA tensors for perturbation forward
    │
    ▼
TempLoRARuntime (in-memory LoRA)
    │
    │ 4. Register plus/minus LoRA slots
    │ 5. Update LoRA tensors each step
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
| TempLoRARuntime | `temp_lora_runtime.py` | In-memory LoRA for perturbation |
| VLLMScorer | `vllm_scorer.py` | Loss computation via vLLM |
| WeightSync | `weight_sync.py` | Sync updated weights to vLLM |

---

## Known Issues

### Single-process mode required

**Issue**: Mock functions only work in single-process mode.

**Environment variables**:
```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0  # Disable multi-process
VLLM_ALLOW_INSECURE_SERIALIZATION=1  # Allow pickle serialization
```

**Performance impact**: ~2-3x slower (CUDA Graphs disabled)

**Future work**: Modify vLLM source to support in-memory LoRA in multi-process mode.

---

## Next Steps

1. Implement complete LOZO training loop (2D params only)
2. Test convergence with SST2 dataset
3. Compare with baseline:
   - Loss curves
   - Accuracy metrics
4. Decide on 1D params handling