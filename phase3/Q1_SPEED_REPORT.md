# Phase 3 q=1 Speed Report

> Historical checkpoint: this report records the initial q=1 speed result that
> exposed vLLM under-utilization at larger batches. It is not the latest Phase 3
> speed checkpoint. See `phase3/STATUS.md` for the direct-worker scaling result.

## Scope

This report freezes the first Phase 3 speed result for a single perturbation
direction (`q=1`). It compares a minimal LOZO baseline against the vLLM training
path after removing avoidable debug overhead.

Phase 2 remains frozen for correctness and convergence validation.

## Configuration

| field | value |
|---|---|
| model | `facebook/opt-2.7b` |
| dataset | GLUE SST-2 train subset |
| steps | `1000` |
| batch size | `16` |
| num samples | `1000` |
| rank | `8` |
| learning rate | `3e-7` |
| epsilon | `1e-3` |
| step interval | `50` |
| eval interval | `0` |
| seed | `42` |
| ZO random device | `cuda` |
| train scope | `lora_only` |
| vLLM batch invariant | `0` |
| vLLM enforce eager | `0` |
| vLLM LoRA residency | `gpu` |
| vLLM LoRA injection | `direct` |
| vLLM weight update | `direct` |
| vLLM weight update precision | `param` |

## Formal Batch Sweep Run

Run directory:

```text
phase3/results/phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live/
```

Artifacts:

```text
phase3/results/phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live/summary.md
phase3/results/phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live/validation.json
```

Validation result:

```text
ok=true
errors=[]
warnings=[]
```

## Batch Sweep Result

| batch | LOZO loss drop | vLLM loss drop | LOZO total s/step | vLLM total s/step | vLLM speedup |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.621094 | 0.624415 | 0.093647 | 0.075339 | 1.2430 |
| 32 | 0.539062 | 0.547690 | 0.115940 | 0.104498 | 1.1095 |
| 64 | 0.519531 | 0.545450 | 0.168678 | 0.165050 | 1.0220 |
| 128 | 0.554688 | 0.565803 | 0.279311 | 0.298030 | 0.9372 |

The speed curve is not monotonic in favor of vLLM for `q=1`. vLLM is faster at
batch 16 and 32, essentially tied at batch 64, and slower at batch 128. The
observed GPU utilization during the run was consistent with this result:
LOZO often drove the assigned GPU close to full utilization, while the vLLM
training path frequently stayed around the 50% SM-utilization range even at
batch 64 and 128.

## Detailed Batch Timing

| batch | score_generate_s | score_generate share | lora_update_s | weight_update_s | build_lora_s | direction_s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.045479 | 61.52% | 0.010884 | 0.007866 | 0.007176 | 0.002326 |
| 64 | 0.135545 | 82.68% | 0.010531 | 0.007933 | 0.006862 | 0.002321 |
| 128 | 0.266845 | 89.90% | 0.010867 | 0.008083 | 0.006954 | 0.002404 |

As batch size increases, the fixed non-score components stay roughly constant,
but `score_generate_s` grows and dominates the step. Request construction and
NLL postprocessing remain negligible; the current bottleneck is the vLLM
`generate(prompt_logprobs=1)` scoring path with per-request LoRA routing.

## Earlier Single-Batch Run

Run directory:

```text
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/
```

Artifacts:

```text
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/lozo_minimal/lozo_perf_minimal_20260523_190857.json
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/vllm_minimal_eager0/vllm_perf_minimal_20260523_190851.json
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/vllm_detailed_eager0/vllm_perf_detailed_20260523_191049.json
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/summary.md
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/validation.json
```

The results directories are intentionally git-ignored. This report records the
accepted numbers and the commands needed to regenerate and validate them.

## Speed Result

| backend | profile | steps | batch | initial | final | loss drop | loss drop/s | total s/step | speedup vs LOZO |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LOZO | minimal | 1000 | 16 | 5.132812 | 4.511719 | 0.621094 | 0.006668 | 0.093145 | 1.0000 |
| vLLM | minimal | 1000 | 16 | 5.132618 | 4.508140 | 0.624478 | 0.008687 | 0.071890 | 1.2957 |
| vLLM | detailed | 1000 | 16 | 5.132618 | 4.510873 | 0.621745 | 0.008733 | 0.071193 | 1.3083 |

Primary comparison metric:

```text
total_s / step
```

`raw_step_s_mean` is not the primary cross-backend metric because the LOZO
trainer times only the low-rank plus/minus step body, while `total_s/step`
includes the full timed training loop.

## Earlier b16 Detailed Timing

| component | mean s/step | share of raw step |
|---|---:|---:|
| score_s | 0.044184 | 62.13% |
| score_generate_s | 0.044025 | 61.90% |
| score_request_build_s | 0.000007 | 0.01% |
| score_postprocess_s | 0.000079 | 0.11% |
| lora_update_s | 0.010190 | 14.33% |
| weight_update_s | 0.007780 | 10.94% |
| build_lora_s | 0.006700 | 9.42% |
| direction_s | 0.002235 | 3.14% |

For batch 16, the q=1 bottleneck is already the vLLM scoring call:

```text
llm.generate(prompt_logprobs=1)
```

Request construction and NLL postprocessing are negligible.

## Batch Sweep Validation

Validation command:

```bash
.venv/bin/python -u phase3/runners/validate_q1_batch_sweep.py \
  phase3/results/phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live \
  --expected-batches 16,32,64,128 \
  --require-detailed-batches 16,64,128 \
  --output phase3/results/phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live/validation.json
```

Validation result:

```text
ok=true
errors=[]
warnings=[]
```

## Earlier b16 Validation

Validation command:

```bash
.venv/bin/python -u phase3/runners/validate_q1_speed_suite.py \
  phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun \
  --min-vllm-speedup 1.2 \
  --output phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/validation.json
```

Validation result:

```text
ok=true
vllm_minimal_speedup=1.2957
vllm_detailed_speedup=1.3083
score_generate_share=61.90%
```

## Reproduction

Use externally assigned GPUs. Do not hard-code GPU IDs in scripts or docs.

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/launch_q1_speed_suite.py \
  --tmux-session zo-vllm \
  --run-id phase3_q1_speed_noeval_b16_s1000_<timestamp> \
  --steps 1000 \
  --batch-size 16 \
  --num-samples 1000 \
  --include-detailed
```

Then collect and validate:

```bash
.venv/bin/python -u phase3/runners/collect_q1_speed_suite.py \
  phase3/results/<run_id> \
  --output phase3/results/<run_id>/summary.md

.venv/bin/python -u phase3/runners/validate_q1_speed_suite.py \
  phase3/results/<run_id> \
  --min-vllm-speedup 1.2 \
  --output phase3/results/<run_id>/validation.json
```

## Conclusion

For `q=1`, vLLM is beneficial only at smaller batch sizes under the current
implementation. The best formal batch-sweep speedup is `1.243x` at batch 16,
then drops to `1.109x` at batch 32, `1.022x` at batch 64, and `0.937x` at batch
128. Loss drop stays comparable or slightly larger for vLLM in this run, so the
negative result is about wall-clock efficiency, not failed convergence.

The next Phase 3 optimization target should be the scoring path itself:
`llm.generate(prompt_logprobs=1)` plus temporary LoRA routing/slot mapping.
Increasing `q=1` batch size alone does not saturate vLLM enough to beat LOZO at
large batch.

Use `phase3/runners/microbench_vllm_scoring.py` before changing the training
algorithm. It keeps `q=1` and splits the scoring path into base
`prompt_logprobs`, fixed-LoRA scoring, slot rewrite without scoring, and the
current rewrite+score path.

Example:

```bash
CUDA_VISIBLE_DEVICES=<GPU> .venv/bin/python -u \
  phase3/runners/microbench_vllm_scoring.py \
  --batch-size 16 \
  --num-samples 1000 \
  --steps 200 \
  --warmup-steps 5 \
  --enforce-eager 1 \
  --modes base,static_lora,rewrite_lora_only,score_with_rewrite \
  --output-dir phase3/results/<run_id>/microbench_b16
```

## Reproducing the q=1 Batch Sweep

Keep `q=1` fixed and sweep batch size:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/launch_q1_batch_sweep.py \
  --tmux-session zo-vllm \
  --batch-sizes 16,32,64,128 \
  --num-samples 1000 \
  --steps 1000
```

Use `--dry-run` first if you only want to inspect commands. The launcher keeps
LOZO and vLLM on separately assigned GPUs and runs each
batch list sequentially per backend. It is intended to test whether the current
vLLM scoring bottleneck improves with larger q=1 training batches before
introducing multiple perturbation directions.

After the tmux windows finish, collect the paired per-batch comparison with:

```bash
.venv/bin/python -u phase3/runners/collect_q1_batch_sweep.py \
  phase3/results/<run_id> \
  --output phase3/results/<run_id>/summary.md
```

Then validate that every batch has paired LOZO/vLLM results with matching
configuration:

```bash
.venv/bin/python -u phase3/runners/validate_q1_batch_sweep.py \
  phase3/results/<run_id> \
  --expected-batches 16,32,64,128 \
  --require-detailed-batches 16,64,128 \
  --output phase3/results/<run_id>/validation.json
```
