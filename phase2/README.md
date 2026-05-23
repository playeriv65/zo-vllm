# Phase 2 Acceptance Summary

This directory contains the strict LOZO baseline vs vLLM fake-LoRA convergence
path for OPT-2.7B. The current accepted configuration is:

| model | backend | rank | step_interval | lr | eps | batch_size |
|---|---|---:|---:|---:|---:|---:|
| `facebook/opt-2.7b` | vLLM fake-LoRA LOZO | 8 | 50 | 3e-7 | 1e-3 | 16 |

## Acceptance Results

| check | result | status |
|---|---:|---|
| baseline/vLLM initial loss diff | 0.000241 | PASS |
| 300-step vLLM loss drop vs baseline | 98.6% | PASS |
| 300-step final loss diff | 0.003750 | PASS |
| 300-step sign match | 96.7% | PASS |
| 300-step high-signal sign match | 97.6% | PASS |
| vLLM training speed | 2.42 steps/s | PASS |
| baseline training speed | 2.03 steps/s | reference |
| CUDA RNG side-by-side U/V digest mismatch | 0 / 20 steps | PASS |
| CUDA RNG side-by-side max plus/minus loss diff | 0.010078 / 0.010059 | PASS |
| CUDA RNG baseline speedup vs CPU RNG | 5.20x on 20-step side-by-side | PASS |
| GPU-resident LoRA side-by-side smoke | 3 / 3 steps accepted | PASS |
| GPU direct injection side-by-side | 20 / 20 steps accepted | PASS |
| GPU direct injection lora_update speedup | 1.37x vs manager | PASS |
| GPU-resident LoRA vs CPU mock short path | exact 3-step vLLM match, 1.16x step speed | PASS |
| baseline full-scope ablation loss drop | -0.250000 vs -0.140625 | reference |
| sample-level batch invariance max NLL diff | 0.000000000 | PASS |
| memory LoRA write/register speedup | 88.1x / 111.9x / 159.4x | PASS |

The formal local result table is generated at
`phase2_results/convergence/official_results.md`. `phase2_results/` is ignored
because it contains run logs and JSON artifacts.

## Main Entry Points

Run one strict comparison:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
.venv/bin/python phase2/run_convergence_experiment.py \
  --backend both \
  --steps 300 \
  --batch-size 16 \
  --rank 8 \
  --lr 3e-7 \
  --eps 1e-3 \
  --step-interval 50 \
  --zo-random-device cuda \
  --lora-residency gpu \
  --lora-injection direct \
  --batch-invariant 0 \
  --enforce-eager 1 \
  --eval-interval 20 \
  --output-dir phase2_results/convergence/manual_r8_si50_lr3e-7 \
  --no-wandb
```

Run the 100-step sweep:

```bash
.venv/bin/python phase2/run_convergence_sweep.py \
  --backend vllm \
  --steps 100 \
  --batch-size 16 \
  --eps 1e-3 \
  --eval-interval 20 \
  --lora-residency gpu \
  --lora-injection direct \
  --batch-invariant 0 \
  --enforce-eager 1 \
  --lrs 1e-7,3e-7,1e-6 \
  --ranks 8,16 \
  --step-intervals 50,100 \
  --output-root phase2_results/convergence/sweep100_manual
```

## Validation Commands

Syntax check:

```bash
.venv/bin/python -m py_compile \
  phase2/lozo_controller.py \
  phase2/temp_lora_runtime.py \
  phase2/vllm_scorer.py \
  phase2/weight_sync.py \
  phase2/run_baseline_helper.py \
  phase2/train_convergence.py \
  phase2/run_convergence_experiment.py \
  phase2/run_convergence_sweep.py \
  phase2/memory_lora_test_utils.py \
  scripts/test_batch_invariance.py
```

GPU validation tests. Pick the currently allocated device outside the script:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> .venv/bin/python phase2/test_real_lozo_baseline_side_by_side.py \
  --steps 3 \
  --eval-interval 3 \
  --zo-random-device cuda \
  --lora-residency gpu \
  --lora-injection direct \
  --batch-invariant 0 \
  --enforce-eager 1 \
  --output-dir phase2_results/convergence/acceptance_side_by_side_smoke

CUDA_VISIBLE_DEVICES=<GPU_IDS> .venv/bin/python scripts/test_batch_invariance.py \
  --output-json phase2_results/batch_invariance/sample_level_manual.json

CUDA_VISIBLE_DEVICES=<GPU_IDS> VLLM_BATCH_INVARIANT=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
.venv/bin/python phase2/test_training_loop.py

CUDA_VISIBLE_DEVICES=<GPU_IDS> .venv/bin/python phase2/test_memory_lora_alignment.py
CUDA_VISIBLE_DEVICES=<GPU_IDS> .venv/bin/python phase2/test_memory_lora_multi.py
CUDA_VISIBLE_DEVICES=<GPU_IDS> .venv/bin/python phase2/test_memory_lora_speed.py
```

## Implementation Notes

- OPT-2.7B must use fp16, not bf16.
- Training defaults to `VLLM_BATCH_INVARIANT=0` for speed. Set it to `1` only
  for explicit sample-level batch-invariance validation or when reproducing the
  older invariant acceptance runs.
- Phase 2 CLIs default to `--lora-residency gpu`. The CPU mock path remains
  available for regression of the original in-memory safetensors loader, but it
  is no longer the speed acceptance path.
- `--lora-injection auto` selects the training-only direct updater on GPU and
  the manager path on CPU. Direct injection creates fixed plus/minus LoRA slots
  once, then overwrites the slot tensors in place each step. `manager` remains
  available as a fallback and comparison path.
- `--enforce-eager 1` is the default because it avoids a large vLLM
  torch.compile/CUDA graph startup cost. `--enforce-eager 0` can improve steady
  training-loop time on longer runs.
- The baseline uses the separate LOZO environment at
  `third_party/LOZO/large_models/.venv` when available.
- `--seed` controls the numpy step-seed stream. `--zo-random-device` controls
  whether LOZO U/V/z tensors are sampled on CPU or CUDA. CUDA sampling is
  the default for Phase 2 CLIs. CPU and CUDA RNG produce different U/V
  sequences, so CPU RNG remains available only for reproducing earlier CPU-RNG
  runs; it is not used for speed acceptance.
- Do not hardcode physical GPU IDs in scripts or docs. Set
  `CUDA_VISIBLE_DEVICES` in the shell/job launcher. The optional `--gpu` flag
  only exists as a one-off environment override and accepts whatever ID string
  the current allocation requires.
- `TempLoRARuntime` uses stable plus/minus LoRA IDs. The CPU mock path still
  uses vLLM `LoRARequest(load_inplace=True)` to avoid stale cache hits. The GPU
  direct path initializes two slots through the manager once, then writes
  `lora_a_stacked`/`lora_b_stacked` through each wrapper's `set_lora()` without
  per-step `remove_adapter/add_adapter/activate_adapter`.
- `LOZOController` supports `train_scope=lora_only` for vLLM-compatible
  Linear-only perturbations and `train_scope=full` for HF baseline ablations
  that include embeddings and 1D parameters.
- `WeightSync` copies packed QKV slices in place instead of cloning the full
  packed tensor per layer.

## Recent Stepwise And Ablation Results

CUDA RNG side-by-side, 20 steps:

```text
steps_compared=20
seed_mismatch_steps=[]
direction_digest_mismatch_steps=[]
loss_fail_steps@0.04=[]
c_fail_steps@25=[]
sign_fail_steps=[]
max_loss_plus_diff=0.010078
max_loss_minus_diff=0.010059
max_c_diff=6.262078
```

CPU RNG and CUDA RNG are not bitwise-equivalent random streams. Comparing the
20-step baseline direction digests across CPU-RNG and CUDA-RNG runs gives
`0/20` matches. CUDA RNG is therefore a separate accepted perturbation stream,
not a drop-in identical replacement for CPU RNG.

RNG speed comparison on the 20-step side-by-side baseline:

| rng_device | step_s_mean | relative |
|---|---:|---:|
| `cpu` | 0.6180 | 1.00x |
| `cuda` | 0.1188 | 5.20x faster |

Baseline train-scope ablation, 100 steps, `rank=8`, `lr=3e-7`,
`eps=1e-3`, `step_interval=50`, `batch_size=16`, CUDA RNG:

| scope | initial | final | loss_change | step_s_mean |
|---|---:|---:|---:|---:|
| `lora_only` | 5.132812 | 4.992188 | -0.140625 | 0.1168 |
| `full` | 5.132812 | 4.882812 | -0.250000 | 0.1340 |

The full baseline, which includes embeddings and 1D parameters, drops loss
faster in this 100-step run, but it is not an order-of-magnitude difference.

GPU-resident LoRA smoke, `rank=8`, `lr=1e-7`, `eps=1e-3`,
`step_interval=100`, `batch_size=2`, CUDA RNG:

| check | result |
|---|---:|
| vLLM CPU mock vs GPU residency seed/digest/loss/c mismatch | 0 / 3 steps |
| CPU mock `step_s_mean` | 0.276578 |
| GPU residency `step_s_mean` | 0.238013 |
| CPU mock `score_s_mean` | 0.108493 |
| GPU residency `score_s_mean` | 0.061117 |
| default GPU side-by-side max plus/minus loss diff vs baseline | 0.007089 / 0.001356 |
| default GPU side-by-side max c diff vs baseline | 2.866773 |

Direct slot updater, 20-step vLLM-only, `rank=8`, `lr=1e-7`, `eps=1e-3`,
`step_interval=100`, `batch_size=16`, CUDA RNG,
`batch_invariant=0,enforce_eager=1`:

| injection | total_s | step_s_mean | tail10_step_s_mean | lora_update_s_mean | tail10_lora_update_s_mean |
|---|---:|---:|---:|---:|---:|
| `manager` | 4.3566 | 0.2144 | 0.2140 | 0.0147 | 0.0150 |
| `direct` | 4.2460 | 0.2092 | 0.2019 | 0.0107 | 0.0108 |

Direct injection preserved the step seed stream and U/V digests. Compared with
the manager path, `lora_update_s_mean` improved by `1.37x` and total loop time
by `1.03x`; the modest total speedup is expected because scoring and weight
sync dominate the step time.

Direct side-by-side vs LOZO baseline, 20 steps:

```text
steps_compared=20
seed_mismatch_steps=[]
direction_digest_mismatch_steps=[]
loss_fail_steps@0.04=[]
c_fail_steps@25=[]
sign_fail_steps=[]
max_loss_plus_diff=0.009428
max_loss_minus_diff=0.009200
max_c_diff=6.066894
```

Execution flag ablation, 20-step vLLM-only, `rank=8`, `lr=1e-7`,
`eps=1e-3`, `step_interval=100`, `batch_size=16`, CUDA RNG,
GPU-resident manager-path LoRA:

| batch_invariant | enforce_eager | total_s | step_s_mean | tail10_step_s_mean | score_s_mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 4.6126 | 0.2273 | 0.2106 | 0.0768 |
| 0 | 0 | 4.2308 | 0.2092 | 0.2001 | 0.0580 |
| 1 | 1 | 4.5978 | 0.2257 | 0.2188 | 0.0754 |
| 1 | 0 | 4.5380 | 0.2235 | 0.2142 | 0.0737 |

All four runs used identical step seeds and U/V direction digests. With
`batch_invariant=0`, disabling eager was the fastest training loop
(`1.09x` tail-10 speedup vs `batch_invariant=1,enforce_eager=1`), but a cached
`batch_invariant=1,enforce_eager=0` run still spent `23.05s` in vLLM engine
initialization (`8.98s` compile plus `11s` CUDA graph capture). For current
short acceptance and 300-step comparisons, `batch_invariant=0,enforce_eager=1`
is the pragmatic default; use `enforce_eager=0` for longer throughput runs.
