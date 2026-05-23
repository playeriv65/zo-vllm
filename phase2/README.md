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
| baseline full-scope ablation loss drop | -0.250000 vs -0.140625 | reference |
| sample-level batch invariance max NLL diff | 0.000000000 | PASS |
| memory LoRA write/register speedup | 88.1x / 111.9x / 159.4x | PASS |

The formal local result table is generated at
`phase2_results/convergence/official_results.md`. `phase2_results/` is ignored
because it contains run logs and JSON artifacts.

## Main Entry Points

Run one strict comparison:

```bash
VLLM_BATCH_INVARIANT=1 \
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
- `VLLM_BATCH_INVARIANT=1` is required for reproducible prompt logprobs.
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
- `TempLoRARuntime` uses stable plus/minus LoRA IDs and vLLM
  `LoRARequest(load_inplace=True)` to avoid stale LoRA cache hits.
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
