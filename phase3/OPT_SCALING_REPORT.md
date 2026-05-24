# Phase 3 OPT Scaling Report

This report records the current Phase 3 q=1 OPT-family throughput checkpoint.
It is intentionally scoped to speed measurement and accounting validation. It
does not replace the Phase 2 correctness and convergence evidence.

## Scope

The experiment compares a minimal LOZO baseline against the optimized vLLM
training path across:

```text
models: facebook/opt-1.3b, facebook/opt-2.7b, facebook/opt-6.7b, facebook/opt-13b
batch sizes: 16, 32, 64, 128
perturbation directions: q=1
```

Run directory:

```text
phase3/results/phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_20260524_current/
```

Collected artifacts:

```text
phase3/results/phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_20260524_current/summary.md
phase3/results/phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_20260524_current/summary.json
phase3/results/phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_20260524_current/validation.json
```

The result directory is git-ignored. This report stores the audited numbers and
the interpretation rules needed to avoid overstating the speedup.

## Measurement Contract

The primary cross-backend metric is:

```text
total_s_per_step = timing.total_s / config.steps
speedup = lozo_total_s_per_step / vllm_total_s_per_step
```

The metric is deliberately based on a direct wall-clock measured training span,
not on the sum of smaller component timers. Component timers are only used to
explain where time goes.

Both runners use:

```text
steps=1000
warmup_steps=5
total executed steps=1005
```

Warmup steps are excluded from `timing.total_s`. The timed span starts at the
first post-warmup measured training step and ends after the last measured
training step update.

LOZO timing includes the full measured training step: plus/minus perturbation
evaluation, low-rank ZO update, optimizer/scheduler update overhead, and
trainer loop overhead. vLLM timing includes direction sampling, direct LoRA slot
write, plus/minus scoring, direct base-weight update, and Python loop overhead.

`step_s_mean` is not the formal cross-backend metric. It is useful for
debugging, but the report uses `timing.total_s / config.steps` consistently.

## Controlled Configuration

Common configuration verified from all 32 JSON files:

| field | value |
|---|---|
| dataset | GLUE SST-2 train subset |
| steps | `1000` |
| warmup steps | `5` |
| num samples | `1000` |
| rank | `8` |
| learning rate | `3e-7` |
| epsilon | `1e-3` |
| step interval | `50` |
| eval interval | `0` |
| seed | `42` |
| ZO random device | `cuda` |
| train scope | `lora_only` |

LOZO configuration:

| field | value |
|---|---|
| profile mode | `minimal` |
| torch compile | `false` |
| torch compile mode | `default` |
| embedding and 1D params | not perturbed (`lora_only`) |

vLLM configuration:

| field | value |
|---|---|
| profile mode | `detailed` |
| enforce eager | `0` |
| batch invariant | `0` |
| scoring backend | `direct_worker` |
| base eval mode | `skip` |
| LoRA residency | `gpu` |
| LoRA injection | `direct` |
| direct LoRA from directions | `1` |
| weight update | `direct` |
| weight update precision | `param` |
| QKV weight update | `batched` |
| sync weight update | `0` |
| direction sampling | `flat` |
| GPU memory utilization | `0.3` for 1.3B/2.7B/6.7B, `0.5` for 13B |

The 13B vLLM runs used higher `gpu_memory_utilization` because the initial 0.3
setting could not allocate enough KV cache. This does not change the ZO math,
but it is a measurement caveat. If a paper table requires one uniform vLLM
engine memory setting across all models, rerun the smaller models with the same
memory setting or rerun 13B with a different shape budget and document that
shape change.

Because `base_eval_mode=skip`, vLLM loss drop is intentionally `n/a` in this
speed table. This checkpoint is not a convergence claim.

## Audited Speed Results

The table below was recomputed from the per-run JSON files using
`timing.total_s / config.steps`.

| model | batch | LOZO s/step | vLLM s/step | vLLM speedup | LOZO loss drop | vLLM loss drop |
|---|---:|---:|---:|---:|---:|---:|
| facebook/opt-1.3b | 16 | 0.051799 | 0.017956 | 2.8847 | 0.726562 | n/a |
| facebook/opt-1.3b | 32 | 0.061504 | 0.024607 | 2.4995 | 0.609375 | n/a |
| facebook/opt-1.3b | 64 | 0.087681 | 0.030194 | 2.9039 | 0.582031 | n/a |
| facebook/opt-1.3b | 128 | 0.145607 | 0.052543 | 2.7712 | 0.628906 | n/a |
| facebook/opt-2.7b | 16 | 0.093252 | 0.027187 | 3.4300 | 0.617188 | n/a |
| facebook/opt-2.7b | 32 | 0.115986 | 0.034143 | 3.3970 | 0.535156 | n/a |
| facebook/opt-2.7b | 64 | 0.168131 | 0.051207 | 3.2833 | 0.519531 | n/a |
| facebook/opt-2.7b | 128 | 0.278699 | 0.093098 | 2.9936 | 0.554688 | n/a |
| facebook/opt-6.7b | 16 | 0.311900 | 0.049019 | 6.3628 | 0.433594 | n/a |
| facebook/opt-6.7b | 32 | 0.366164 | 0.068356 | 5.3568 | 0.433594 | n/a |
| facebook/opt-6.7b | 64 | 0.483211 | 0.110305 | 4.3807 | 0.375000 | n/a |
| facebook/opt-6.7b | 128 | 0.730853 | 0.196829 | 3.7131 | 0.421875 | n/a |
| facebook/opt-13b | 16 | 0.630199 | 0.088202 | 7.1449 | -0.988281 | n/a |
| facebook/opt-13b | 32 | 0.731524 | 0.122868 | 5.9538 | -0.742188 | n/a |
| facebook/opt-13b | 64 | 0.959373 | 0.205176 | 4.6759 | -0.925781 | n/a |
| facebook/opt-13b | 128 | 1.458094 | 0.367386 | 3.9688 | -0.777344 | n/a |

The 13B LOZO loss increases under this hyperparameter setting. Those 13B rows
are valid throughput measurements, but they are not convergence evidence.

## vLLM Timing and Utilization

vLLM component means nearly add up to `total_s_per_step`, so the vLLM timing
breakdown has no large hidden gap. The remaining cost is dominated by scoring,
especially as model and batch size increase.

| model | batch | vLLM s/step | score s | score share | GPU util mean | GPU util median |
|---|---:|---:|---:|---:|---:|---:|
| facebook/opt-1.3b | 16 | 0.017956 | 0.014224 | 79.21% | 56.86 | 66.50 |
| facebook/opt-1.3b | 32 | 0.024607 | 0.020752 | 84.33% | 59.81 | 71.00 |
| facebook/opt-1.3b | 64 | 0.030194 | 0.026225 | 86.85% | 76.76 | 90.00 |
| facebook/opt-1.3b | 128 | 0.052543 | 0.048378 | 92.07% | 86.16 | 92.00 |
| facebook/opt-2.7b | 16 | 0.027187 | 0.022302 | 82.03% | 68.54 | 80.00 |
| facebook/opt-2.7b | 32 | 0.034143 | 0.029146 | 85.36% | 74.94 | 89.00 |
| facebook/opt-2.7b | 64 | 0.051207 | 0.045959 | 89.75% | 86.55 | 96.00 |
| facebook/opt-2.7b | 128 | 0.093098 | 0.087407 | 93.89% | 96.86 | 97.00 |
| facebook/opt-6.7b | 16 | 0.049019 | 0.043647 | 89.04% | 58.60 | 95.50 |
| facebook/opt-6.7b | 32 | 0.068356 | 0.062981 | 92.14% | 96.42 | 97.00 |
| facebook/opt-6.7b | 64 | 0.110305 | 0.104612 | 94.84% | 97.39 | 97.00 |
| facebook/opt-6.7b | 128 | 0.196829 | 0.190978 | 97.03% | 98.48 | 99.00 |
| facebook/opt-13b | 16 | 0.088202 | 0.081390 | 92.28% | 97.16 | 97.00 |
| facebook/opt-13b | 32 | 0.122868 | 0.115872 | 94.31% | 97.55 | 97.00 |
| facebook/opt-13b | 64 | 0.205176 | 0.198010 | 96.51% | 98.35 | 98.00 |
| facebook/opt-13b | 128 | 0.367386 | 0.360160 | 98.03% | 99.07 | 100.00 |

This utilization profile is materially different from the earlier
under-utilized vLLM path. For large models and batches, the optimized direct
worker path now drives the GPU close to full sampled utilization.

## Accounting Audit

Independent validation over the final run directory found:

```text
result JSON files: 32
expected model/batch/backend pairs: 4 models * 4 batches * 2 backends = 32
duplicate or missing pairs: none
common config mismatches: none for steps, warmup_steps, num_samples, rank, lr,
  eps, step_interval, eval_interval, seed, zo_random_device, train_scope
```

Validation command:

```bash
.venv/bin/python -u phase3/runners/validate_scaling_sweep.py \
  phase3/results/phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_20260524_current \
  --output phase3/results/phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_20260524_current/validation.json
```

Validation result:

```text
ok=true
errors=[]
```

The collector uses the same metric:

```text
phase3/runners/collect_q1_speed_suite.py: total_per_step(data)
phase3/runners/collect_scaling_sweep.py: speedup = lozo_step / vllm_step
```

The runner timing code also matches the accounting rule:

```text
LOZO: measured_train_t0 is set at the first post-warmup step; total_s is
      measured_train_t1 - measured_train_t0 after the final measured update.
vLLM: the same post-warmup measured span is used; final base eval is skipped
      in this speed run.
```

One manifest caveat: the top-level `manifest.json` was updated by resume
launches and no longer represents the full final sweep by itself. Use the
per-result JSON configs and `summary.json` as the source of truth for the final
table. The manifest remains useful for launch history, but not as the complete
final configuration record.

## Interpretation

The large speedup is plausible under this specific throughput accounting, but
it should be described precisely:

- vLLM still evaluates plus and minus perturbations. This result does not claim
  that one mathematical forward computes both signs.
- The speedup comes from the optimized training path: direct worker scoring,
  fixed GPU LoRA slots, direct LoRA writes from U/V directions, direct
  base-weight updates, batched packed-QKV updates, disabled batch-invariant
  mode, and non-eager vLLM execution.
- The comparison baseline is LOZO minimal without `torch.compile`. Do not cite
  this table as a compiled-baseline comparison.
- The comparison baseline and vLLM both use `lora_only`; embeddings and 1D
  parameters are not perturbed.
- Because vLLM base loss evaluation is skipped, this table should be cited as
  `q=1 training-step throughput`, not as loss-drop-per-second or
  time-to-target-loss evidence.

## Reproduction

Use externally assigned GPUs. Do not hard-code GPU IDs in scripts or docs.

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/launch_scaling_sweep.py \
  --tmux-session zo-vllm \
  --run-id phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_<timestamp> \
  --models facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,facebook/opt-13b \
  --batch-sizes 16,32,64,128 \
  --steps 1000 \
  --warmup-steps 5 \
  --num-samples 1000 \
  --backend both
```

Collect the table:

```bash
.venv/bin/python -u phase3/runners/collect_scaling_sweep.py \
  phase3/results/<run_id> \
  --output phase3/results/<run_id>/summary.md \
  --json-output phase3/results/<run_id>/summary.json
```

Validate the accounting contract:

```bash
.venv/bin/python -u phase3/runners/validate_scaling_sweep.py \
  phase3/results/<run_id> \
  --output phase3/results/<run_id>/validation.json
```

When resuming a partial sweep, use `--resume-existing` so completed backend JSON
files are not overwritten. After a resumed run, verify the final table from
per-result JSON files rather than relying only on the top-level manifest.
