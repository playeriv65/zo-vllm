# Phase 2 Acceptance Summary

This directory contains the strict LOZO baseline vs vLLM fake-LoRA convergence
path for OPT-2.7B. The current accepted configuration is:

| model | backend | rank | nu | lr | eps | batch_size |
|---|---|---:|---:|---:|---:|---:|
| `facebook/opt-2.7b` | vLLM fake-LoRA LOZO | 8 | 50 | 3e-7 | 1e-3 | 16 |

## Acceptance Results

Clean-run artifacts were regenerated after clearing `phase1/results`,
`phase1/artifacts`, `phase2/results`, and `phase2/artifacts`.

| check | clean-run result | status |
|---|---:|---|
| 20-step strict side-by-side accepted steps | 20 / 20 | PASS |
| 20-step seed mismatch | 0 / 20 | PASS |
| 20-step U/V digest mismatch | 0 / 20 | PASS |
| 20-step max plus/minus loss diff | 0.030806 / 0.023877 | PASS |
| 20-step max c diff | 18.560467 | PASS |
| 300-step baseline loss | 5.132812 -> 4.832031 | reference |
| 300-step vLLM loss | 5.132858 -> 4.831391 | PASS |
| 300-step vLLM loss drop vs baseline | 100.2% | PASS |
| 300-step final loss diff | 0.000640 | PASS |
| 300-step sign match / high-signal sign match | 98.3% / 99.0% | reference |
| 300-step vLLM step time | 0.0864 s/step | PASS |
| 300-step baseline step time | 0.1155 s/step | reference |
| sample-level batch invariance max NLL diff | 0.000000000 | PASS |
| archived memory LoRA file vs in-memory sign match | 16 / 16 | historical |
| archived memory LoRA file vs in-memory max abs c error | 0.000000 | historical |

The 300-step convergence run is a convergence/speed acceptance, not a strict
per-step tolerance acceptance. It had 6 loss-diff tolerance excursions and 3
c-diff tolerance excursions under the side-by-side thresholds, while preserving
the seed stream and reaching essentially identical final loss.

The formal local result table is generated at
`phase2/results/convergence/official_results.md`. `phase2/results/` is ignored
because it contains run logs and JSON artifacts.

Clean-run validation artifacts:
- `phase2/results/convergence/side_by_side20_20260523_clean/summary.json`
- `phase2/results/convergence/convergence300_20260523_clean/summary.json`
- `phase2/results/batch_invariance/sample_level_20260523_clean.json`
- `phase2/results/milestone1_2_20260523_162450.json` (archived CPU-mock
  comparison; the CPU-mock implementation has since been removed)

## Main Entry Points

Run one strict comparison:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
.venv/bin/python phase2/runners/run_convergence_experiment.py \
  --backend both \
  --steps 300 \
  --batch-size 16 \
  --rank 8 \
  --lr 3e-7 \
  --eps 1e-3 \
  --nu 50 \
  --zo-random-device cuda \
  --weight-update direct \
  --weight-update-precision param \
  --enforce-eager 1 \
  --eval-interval 20 \
  --output-dir phase2/results/convergence/manual_r8_nu50_lr3e-7 \
  --no-wandb
```

Run the 100-step sweep:

```bash
.venv/bin/python phase2/runners/run_convergence_sweep.py \
  --backend vllm \
  --steps 100 \
  --batch-size 16 \
  --eps 1e-3 \
  --eval-interval 20 \
  --weight-update direct \
  --weight-update-precision param \
  --enforce-eager 1 \
  --lrs 1e-7,3e-7,1e-6 \
  --ranks 8,16 \
  --nus 50,100 \
  --output-root phase2/results/convergence/sweep100_manual
```

## Validation Commands

Syntax check:

```bash
.venv/bin/python -m py_compile \
  zo_vllm/training/direction/ \
  zo_vllm/core/lora_runtime/ \
  zo_vllm/experiment/scoring/generate_scorer.py \
  zo_vllm/core/weight_sync.py \
  phase2/runners/run_baseline_helper.py \
  phase2/runners/train_convergence.py \
  phase2/runners/run_convergence_experiment.py \
  phase2/runners/run_convergence_sweep.py \
  phase2/validation/test_batch_invariance.py
```

GPU validation tests. Pick the currently allocated device outside the script:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> .venv/bin/python phase2/validation/test_real_lozo_baseline_side_by_side.py \
  --steps 3 \
  --eval-interval 3 \
  --zo-random-device cuda \
  --weight-update direct \
  --weight-update-precision param \
  --enforce-eager 1 \
  --output-dir phase2/results/convergence/acceptance_side_by_side_smoke

CUDA_VISIBLE_DEVICES=<GPU_IDS> .venv/bin/python phase2/validation/test_batch_invariance.py \
  --output-json phase2/results/batch_invariance/sample_level_manual.json

CUDA_VISIBLE_DEVICES=<GPU_IDS> VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
.venv/bin/python phase2/validation/test_training_loop.py

```

## Implementation Notes

- OPT-2.7B must use fp16, not bf16.
- Training CLIs do not read or set `VLLM_BATCH_INVARIANT`. Dedicated
  sample-level batch-invariance validation scripts handle that env var
  themselves.
- LoRA perturbations use GPU direct slots only. The old CPU mock safetensors
  loader and manager reload comparison path have been removed from active CLIs.
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
- `LoRAUpdateRuntime` uses stable plus/minus LoRA IDs. The direct path
  initializes two slots through the manager once, then writes
  `lora_a_stacked`/`lora_b_stacked` through each wrapper's `set_lora()` without
  per-step `remove_adapter/add_adapter/activate_adapter`.
- `LOZO direction provider` supports only `train_scope=lora_normal` and keeps metadata,
  not full master weight copies.
- `--weight-update direct` applies the LOZO base-weight update inside the vLLM
  worker, so vLLM's base weights become the training master state.

## Clean-Run Details

Strict side-by-side, 20 steps:

```text
steps_compared=20
seed_mismatch_steps=[]
direction_digest_mismatch_steps=[]
loss_fail_steps@0.04=[]
c_fail_steps@25=[]
sign_fail_steps=[]
max_loss_plus_diff=0.030806
max_loss_minus_diff=0.023877
max_c_diff=18.560467
```

Convergence and timing, 300 steps:

| backend | initial | final | loss_change | step_s_mean |
|---|---:|---:|---:|---:|
| baseline | 5.132812 | 4.832031 | -0.300781 | 0.1155 |
| vLLM | 5.132858 | 4.831391 | -0.301467 | 0.0864 |

The clean-run vLLM loop is `1.34x` faster than the instrumented baseline in
this 300-step comparison. Within vLLM, `score_s_mean=0.0585s` dominates the
step time; `weight_update_s_mean=0.0078s`, so the direct base-weight update is
not the current bottleneck.
