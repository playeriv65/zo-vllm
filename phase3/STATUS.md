# Phase 3 Status

## Current Checkpoint

Phase 3 now has one complete OPT-family q=1 throughput checkpoint:

```text
phase3/results/phase3_scaling_models_1.3b-2.7b-6.7b-13b_b16-32-64-128_s300_20260604_191139/
```

Retained tracked data:

```text
phase3/OPT_SCALING_SUMMARY.json
```

The checkpoint compares LOZO minimal against optimized vLLM direct-worker
training for `facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b`, and
`facebook/opt-13b` at batch sizes `16,32,64,128`.

Validation:

```text
ok=true
errors=[]
warnings=[]
```

## Measurement Contract

Current Phase 3 speed runs use:

```text
steps=300
tail_steps=100
batch_size=16
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

The comparable speed metric is the average full step time over the final 100
training steps; the first 200 steps are enough natural warmup for Phase 3.
Older 1000-step reports remain historical context only.
Keep data order fixed across Phase 3 speed comparisons. Training rows use HF
`Dataset.shuffle(data_seed)` and every epoch uses HF `SeedableRandomSampler`
ordering with `dataloader_seed + epoch`. The legacy runner's HF comparison mode
must resolve to the same rule.

All Phase 3 speed reports must keep this block fixed unless the stated ablation
variable is one of these fields. Mixed-parameter comparisons, especially
different `nu`, `rank`, `seed`, sampler, or data-order settings, are invalid for
Phase 3 speed claims.

## Headline Results

| model | b16 | b32 | b64 | b128 |
|---|---:|---:|---:|---:|
| facebook/opt-1.3b | 2.7479x | 2.7762x | 3.0783x | 2.9108x |
| facebook/opt-2.7b | 2.6949x | 3.5483x | 3.2037x | 3.1301x |
| facebook/opt-6.7b | 6.6295x | 4.9530x | 4.1925x | 3.7568x |
| facebook/opt-13b | 7.7993x | 5.5809x | 4.5164x | 3.9788x |

Speedups range from `2.6949x` to `7.7993x`.

## Timing Attribution

Across all 16 rows, the saved time comes from:

| source | contribution |
|---|---:|
| probe/forward/scoring | 89.34% |
| update | 5.74% |
| other | 4.92% |

For `facebook/opt-13b`, absolute saved time grows with batch size:

| batch | saved s/step | probe contrib | update contrib | other contrib |
|---:|---:|---:|---:|---:|
| 16 | 0.633021 | 74.67% | 10.97% | 14.36% |
| 32 | 0.788460 | 79.84% | 8.81% | 11.35% |
| 64 | 1.171238 | 86.93% | 5.91% | 7.16% |
| 128 | 1.979751 | 92.51% | 3.55% | 3.95% |

The main speedup source is direct-worker plus/minus probe scoring. Direct LoRA
slot writes and direct base-weight updates are useful but secondary.

## HF Trainer Same-Parameter Check

The old `+0.901 ms/step` result is invalid: the selected 1000-row datasets only
overlapped by 15 rows and the epoch samplers diverged after epoch one. After
aligning both paths to HF dataset and `SeedableRandomSampler` semantics, the
same-GPU-6 uninstrumented tail means are:

| path | tail step |
|---|---:|
| legacy HF-like | 91.541 ms |
| HF `ZOTrainer` | 91.537 ms |

That historical difference is `-0.004 ms/step`. Its fine-grained HF profile
contained `1.741 ms` of adapter work, including a redundant list-to-tensor-to-list
round trip. The row-object path and second model forward have now been deleted.

The final same-GPU ABBA comparison holds all code except the Trainer/modeling
wrapper fixed. Old-wrapper tail means are `92.351` and `92.211 ms/step`; current
tensor-only HF tail means are `90.831` and `91.358 ms/step`. The mean improvement
is `1.187 ms/step` (`92.281 -> 91.094 ms`, `1.013x`). Both paths use the same
classification probe loss. The current scorer spends `84.138 ms` in vLLM,
`0.320 ms` unpacking the CPU HF tensor batch, `0.376 ms` computing grouped HF
loss, `0.078 ms` postprocessing output, and `0.030 ms` adapting the result. The
generic HF Trainer boundary outside `ZOStepper` is `1.110 ms`.

The current default path is ragged and performs no HF sequence padding. Fixed
OPT-13B reruns on two GPUs measured `90.590 ms` and `91.212 ms` over the final
100 steps (`90.901 ms` mean), with train losses `0.82152` and `0.82178`.
Ragged batch unpacking is `0.015 ms`; vLLM engine scoring is `84.064/84.640 ms`.
The full artifacts are in
`phase3/results/hf_ragged_opt13b_b16_s300_20260710_081316/`.

## Runtime Rules

- Do not hard-code GPU IDs in code or docs.
- Use externally assigned GPUs through launcher arguments or environment
  variables.
- Keep generated outputs under `phase3/results/`.
- Keep `phase3/results/` git-ignored.
- Do not commit result JSON or logs from ignored run directories.
- Treat `phase3/OPT_SCALING_SUMMARY.json` as historical OPT-family checkpoint
  data.
- Use `phase3/runners/launch_superglue_scaling_sweep.py` for current Phase 3
  SST-2-first speed and scaling runs.
