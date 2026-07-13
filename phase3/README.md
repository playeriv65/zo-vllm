# Phase 3: Performance Optimization and Testing

Phase 3 starts from the Phase 2 correctness result and focuses on wall-clock
training efficiency for q=1 zero-order training.

## Current Source of Truth

The historical OPT-family Phase 3 checkpoint is:

```text
phase3/results/phase3_scaling_models_1.3b-2.7b-6.7b-13b_b16-32-64-128_s300_20260604_191139/
```

Because `phase3/results/` is git-ignored, the retained tracked summary for that
historical checkpoint is:

```text
phase3/OPT_SCALING_SUMMARY.json
phase3/STATUS.md
```

The checkpoint compares LOZO minimal against optimized vLLM direct-worker
training for:

```text
models=facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,facebook/opt-13b
batch_sizes=16,32,64,128
```

Current speed runs use:

```text
steps=300
tail_steps=100
batch_size=16
num_train=1000
num_dev=500
rank=2
nu=50
lr=3e-7
eps=1e-3
metric=timing.tail_100.step_s.mean
seed=42
data_seed=42
dataset_selection=hf_dataset_shuffle
train_sampler=hf_seedable_random
dataloader_seed=42
dataloader_drop_last=0
direct_update_mode=accumulate
enforce_eager=0
max_model_len=2048
max_num_batched_tokens=16384
```

Phase 3 does not use an extra warmup-step parameter. The 300-step run is the
unit of work, and the comparable timing metric uses only the final 100 steps.
Keep the data order fixed across Phase 3 speed comparisons. `seed` controls ZO
randomness. Training rows use Hugging Face `Dataset.shuffle(data_seed)`, and the
sampler uses the native `SeedableRandomSampler` rule `dataloader_seed + epoch`.
The legacy runner's `task_shuffle_impl=hf` and `train_sampler=hf_random` options
must resolve to these same semantics.

Any Phase 3 speed comparison must keep this block fixed unless the named
ablation variable is one of these fields. Reports must list the full block
before comparing timings. Do not compare runs with different `nu`, `rank`,
`seed`, sampler, or data-order settings.

The run validated with:

```text
ok=true
errors=[]
warnings=[]
```

## Scope

Phase 2 remains the correctness and convergence baseline. Phase 3 runners live
under `phase3/runners/` only as thin experiment entrypoints and should not
duplicate `zo_vllm` library orchestration.

Phase 3 output directories are local artifacts:

```text
phase3/results/<run_id>/
```

They are ignored by git. Commit only compact reports or curated summary data.

## Active Entry Points

| Name | Script | Purpose |
|---|---|---|
| `phase3-scaling-sweep` | `phase3/runners/launch_superglue_scaling_sweep.py` | Phase 3 SST-2-first speed/scaling matrix defaults and tmux launch. |
| `queue-worker` | `python -m zo_vllm.experiment.runners.job_queue_worker` | Shared dynamic GPU queue worker. |
| `phase3-scaling-collector` | `python -m zo_vllm.experiment.runners.collect_superglue_scaling` | Shared Phase 3 scaling summary collector. |
| `vllm-train` | `python -m zo_vllm.experiment.runners.vllm_zo_task` | vLLM plus/minus direct-worker scoring and direct update timing. |
| `hf-trainer-speed-migration` | `phase3/runners/hf_trainer_speed_migration.py` | Phase-local Hugging Face `ZOTrainer` speed migration probe. |
| `hf-trainer-target-lm-alignment` | `phase3/runners/hf_trainer_target_lm_alignment.py` | Phase-local numeric alignment between legacy `VLLMZOTrainer` and Hugging Face `ZOTrainer` on target-LM data. |

Legacy q=1 and OPT-only Phase 3 runner scripts and reports were removed from
this directory. New experiments should use the shared `zo_vllm.experiment`
infrastructure.

## Current Results

| model | b16 | b32 | b64 | b128 |
|---|---:|---:|---:|---:|
| facebook/opt-1.3b | 2.7479x | 2.7762x | 3.0783x | 2.9108x |
| facebook/opt-2.7b | 2.6949x | 3.5483x | 3.2037x | 3.1301x |
| facebook/opt-6.7b | 6.6295x | 4.9530x | 4.1925x | 3.7568x |
| facebook/opt-13b | 7.7993x | 5.5809x | 4.5164x | 3.9788x |

Across all 16 rows, saved time is attributed to:

| source | contribution |
|---|---:|
| probe/forward/scoring | 89.34% |
| update | 5.74% |
| other | 4.92% |

The dominant gain is direct-worker plus/minus probe scoring. Direct LoRA slot
writes and direct base-weight updates contribute smaller stable savings.

## HF Trainer Same-Parameter Check

The canonical scaling queue marks vLLM jobs with `vllm_runner=hf_phase3`.
Those jobs run the registered SST-2 or SuperGLUE objective through `ZOTrainer`;
the official LOZO comparison backend is unchanged.

Current OPT-13B SST-2 classification wrapper comparison uses identical HF
dataset selection, epoch sampler order, ZO seeds, and runtime parameters. The
old `+0.901 ms/step` comparison is invalid because its two selected 1000-row
training sets overlapped by only 15 rows and its epoch sampler semantics differed.

On GPU 6, the uninstrumented full-step results were `91.541 ms` for the legacy
entrypoint and `91.537 ms` for `ZOTrainer`, a `-0.004 ms` difference. The
instrumented HF rerun measured `91.846 ms` and split its `85.423 ms`
`score_token_groups` interval into:

| HF scoring component | tail mean |
|---|---:|
| vLLM engine scoring | 83.682 ms |
| list to GPU tensor batch build | 0.819 ms |
| GPU tensor to token-list unpack | 0.466 ms |
| classification loss postprocess | 0.297 ms |
| result adaptation | 0.002 ms |
| remaining dispatch and input preparation | 0.157 ms |

This historical HF adapter contributed `1.741 ms` inside the scoring callback.
It was mainly a redundant list-to-tensor-to-list round trip. That path has now
been deleted: Trainer computes loss directly from compact vLLM outputs without
rebuilding a probe tensor batch or invoking model forward again.

The final same-GPU ABBA check restored only the deleted Trainer/modeling wrapper
in a temporary worktree while keeping data, sampler, vLLM, update code, and all
runtime parameters identical. The two old-wrapper tail means were `92.351` and
`92.211 ms/step`; the two tensor-only HF results were `90.831` and
`91.358 ms/step`. Their means are `92.281` and `91.094 ms/step`, so deleting the
redundant wrapper saves `1.187 ms/step` (`1.013x`). Both paths report the same
classification training objective (`0.8217` mean train loss).

The current scorer mean is `84.642 ms`: `84.138 ms` in vLLM engine scoring,
`0.320 ms` unpacking the CPU HF tensor batch, `0.376 ms` computing one HF loss
per perturbation group, `0.078 ms` output postprocessing, and `0.030 ms` result
adaptation. The generic HF Trainer boundary outside `ZOStepper` is `1.110 ms`.
The controlled artifacts are under
`phase3/results/hf_wrapper_abba_gpu6_20260710_065008/`.

The subsequent ragged-collator implementation removes sequence padding and the
CPU tensor unpack from the default path. Two fixed-contract OPT-13B runs measured
`90.590 ms` and `91.212 ms` per tail step, with a `90.901 ms` mean. This is
`0.193 ms` faster than the tensor-only ABBA mean. Their train losses were
`0.82152` and `0.82178`. Tail profile means were:

| component | GPU 6 | GPU 7 |
|---|---:|---:|
| full step | 90.590 ms | 91.212 ms |
| vLLM engine scoring | 84.064 ms | 84.640 ms |
| ragged batch unpack | 0.015 ms | 0.015 ms |
| classification output assembly | 0.493 ms | 0.492 ms |
| HF grouped loss | 0.114 ms | 0.113 ms |
| result adaptation | 0.105 ms | 0.108 ms |

Artifacts are under
`phase3/results/hf_ragged_opt13b_b16_s300_20260710_081316/`.

## Reproduction

Use externally assigned GPUs. Do not hard-code GPU IDs in scripts or docs.

```bash
.venv/bin/python -u phase3/runners/launch_superglue_scaling_sweep.py \
  --gpus <GPU_IDS> \
  --tasks sst2,superglue_boolq,superglue_cb,superglue_copa,superglue_multirc,superglue_record,superglue_rte,superglue_wic,superglue_wsc \
  --models facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,facebook/opt-13b \
  --batch-sizes 16,32,64,128 \
  --steps 300 \
  --num-samples 1000 \
  --num-dev 500 \
  --rank 2 \
  --lr 3e-7 \
  --eps 1e-3 \
  --nu 50 \
  --eval-interval 300 \
  --seed 42 \
  --train-set-seed 42 \
  --train-sampler sequential \
  --dataloader-seed 42 \
  --gpu-memory-utilization 0.9 \
  --direction-sampling flat \
  --direct-update-mode accumulate \
  --qkv-weight-update batched \
  --tmux-session zo-vllm
```

Collect:

```bash
.venv/bin/python -u -m zo_vllm.experiment.runners.collect_superglue_scaling \
  phase3/results/<run_id> \
  --output phase3/results/<run_id>/summary.md \
  --json-output phase3/results/<run_id>/summary.json
```

HF Trainer migration speed probe:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
.venv/bin/python -u phase3/runners/hf_trainer_speed_migration.py \
  --run-id <run_id> \
  --model facebook/opt-13b \
  --task sst2 \
  --task-objective sst2_classification \
  --steps 300 \
  --tail-steps 100 \
  --batch-size 16 \
  --dataset-repeats 1 \
  --rank 2 \
  --nu 50 \
  --lr 3e-7 \
  --eps 1e-3 \
  --seed 42 \
  --data-seed 42 \
  --max-length 2048 \
  --max-model-len 2048 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 0 \
  --gpu-memory-utilization 0.5
```

This probe uses the generic `zo_trainer.ZOTrainer` path and Hugging Face train
metrics. Phase 3-specific tail-window timing is computed inside this phase
script and written to `phase3/results/<run_id>/result.json`; it is not part of
the generic `zo_trainer` API.
This probe only accepts HF-native token batches produced by Hugging Face
`Dataset.map`; it does not support raw-row task objectives or legacy
`TokenProbeBatch` collation.

HF Trainer target-LM numeric alignment:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
.venv/bin/python -u phase3/runners/hf_trainer_target_lm_alignment.py \
  --run-id <run_id> \
  --model facebook/opt-125m \
  --steps 10 \
  --eval-steps 5 \
  --batch-size 2 \
  --data-seed 42 \
  --rank 1 \
  --nu 10 \
  --lr 1e-7 \
  --eps 1e-3
```

This check runs the legacy trainer and Hugging Face `ZOTrainer` in isolated
child processes with the same tokenized target-LM rows, perturbation settings,
and evaluation cadence. It writes `legacy_result.json`, `hf_result.json`, and
`alignment_result.json` under `phase3/results/<run_id>/`.

## LOZO Fast Baseline Alignment Check

Before using `LOZO_FAST_MODE=1` as a baseline timing path, run the small OPT-125M
alignment check:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
third_party/LOZO/large_models/.venv/bin/python -u \
  phase3/runners/test_lozo_lora_backend_alignment.py \
  --model-name facebook/opt-125m \
  --batch-size 2 \
  --seq-len 32 \
  --steps 4 \
  --rank-r 2 \
  --step-interval 2 \
  --output-json phase3/results/<run_id>/alignment.json
```

The check compares `dense/dense`, `lora/dense`, and `lora/lazy_lora` with exact
default tolerances for the LOZO seed stream, direction RNG digests, plus/minus
losses, projected gradients, and post-update effective clean loss. The
`--step-interval 2 --steps 4` setting also covers a cached-`V` reuse step and a
lazy-update fold step.

## Acceptance Gates

- Phase 3 scripts do not duplicate shared `zo_vllm.experiment` orchestration.
- Minimal LOZO and minimal vLLM runs produce JSON with config and timing.
- LOZO fast baseline timing claims pass the OPT-125M alignment check above.
- A valid comparison report states the timing metric, model/batch grid, and
  whether vLLM evaluation was skipped.
- Timing-attribution claims are backed by aligned phase timers, not by raw
  speedup alone.
