# Phase 3: Performance Optimization and Testing

Phase 3 starts from the Phase 2 correctness result and focuses on wall-clock
training efficiency.

## Current Checkpoint

Phase 3 is still in progress. The latest formal throughput checkpoint is the
OPT-family q=1 scaling run documented in `phase3/OPT_SCALING_REPORT.md`. It
compares LOZO minimal against the optimized vLLM direct-worker path for
`facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b`, and
`facebook/opt-13b` at batch sizes `16,32,64,128`.

That report is the source of truth for the current speedup accounting. It also
records the caveats: vLLM base evaluation was skipped, the LOZO baseline was not
compiled with `torch.compile`, both sides use `lora_only`, and the 13B vLLM rows
required a higher vLLM GPU memory utilization setting.

The current code captures a checkpoint after fixing several obvious q=1
throughput problems:

- direct worker scoring for prompt token IDs
- direct plus/minus LoRA slot writes from U/V directions
- optional warmup steps excluded from measured timing
- corrected LOZO wall-clock step timing that includes update overhead
- flat GPU direction sampling and batched packed-QKV weight updates

Latest formal checkpoint artifacts are under:

```text
phase3/results/phase3_scaling_opt1p3b_opt2p7b_opt6p7b_opt13b_b16_32_64_128_s1000_w5_20260524_current/
```

The result directory is git-ignored; the key summary is recorded in
`phase3/OPT_SCALING_REPORT.md` and `phase3/STATUS.md`. These speed runs use
`base_eval_mode=skip` on the vLLM side, so they are throughput checkpoints
rather than Phase 2-style convergence acceptance runs. New `facebook/opt-1.3b`
runs should use the `opt1p3b` slug.

## Scope

Phase 2 runners stay frozen for alignment and convergence validation. Phase 3
adds separate performance runners under `phase3/runners/` so speed experiments
do not change the accepted Phase 2 evidence.

## Baseline Definitions

| Name | Script | Purpose |
|---|---|---|
| `lozo-minimal` | `phase3/runners/train_lozo_baseline_perf.py --profile-mode minimal` | LOZO plus/minus-only timing with minimal per-step logging. |
| `lozo-instrumented` | `phase3/runners/train_lozo_baseline_perf.py --profile-mode instrumented` | LOZO timing with base loss, direction digest, and per-step history for debugging. |
| `vllm-minimal` | `phase3/runners/train_vllm_perf.py --profile-mode minimal` | vLLM plus/minus scoring and direct update timing with minimal per-step logging. |
| `vllm-detailed` | `phase3/runners/train_vllm_perf.py --profile-mode detailed` | vLLM timing with score request/generate/postprocess breakdown. |
| `vllm-scoring-microbench` | `phase3/runners/microbench_vllm_scoring.py` | Isolates q=1 base scoring, fixed-LoRA scoring, slot rewrite, and current rewrite+score timing. |
| `scaling-sweep` | `phase3/runners/launch_scaling_sweep.py` | Runs LOZO and optimized vLLM across OPT models and batch sizes. |
| `scaling-validator` | `phase3/runners/validate_scaling_sweep.py` | Checks the OPT scaling accounting contract and per-result JSON configs. |

The default comparable speed pair is:

```text
lozo-minimal vs vllm-minimal
```

The default alignment/debug pair is:

```text
lozo-instrumented vs vllm-detailed
```

## Current Performance Question

The first Phase 3 question is whether vLLM remains faster than LOZO after both
sides remove avoidable debug overhead, while keeping the perturbation count at
`q=1`.

The controlled profiling configuration is:

```text
model=facebook/opt-1.3b or facebook/opt-2.7b
dataset=GLUE SST-2 train subset
steps=1000
batch_size=16
num_samples=1000
rank=8
lr=3e-7
eps=1e-3
step_interval=50
seed=42
zo_random_device=cuda
train_scope=lora_only
```

vLLM default:

```text
lora_residency=gpu
lora_injection=direct
weight_update=direct
weight_update_precision=param
batch_invariant=0
enforce_eager=0
direction_digest=off
```

## Current q=1 Results

The latest speed checkpoint is summarized in `phase3/STATUS.md`. The older q=1
results below are retained because they document the bottleneck that motivated
direct worker scoring.

Formal q=1 speed runs use the same configuration above with `eval_interval=0`.
Initial and final losses are still computed outside the timed training loop.

Run artifacts:

```text
local ignored vLLM run ending in 20260524_123159
local ignored LOZO run ending in 20260524_121646
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_185158/
phase3/results/phase3_q1_detailed_noeval_b16_s1000_20260523_185514/
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/
phase3/results/phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live/
```

Latest OPT-1.3B batch-16 checkpoint:

| backend | total s/step | speedup vs LOZO | notes |
|---|---:|---:|---|
| LOZO minimal | 0.051668 | 1.00x | corrected full wall-clock step |
| vLLM detailed | 0.017034 | 3.03x | direct worker score, direct slot/update path |

Latest vLLM breakdown:

| component | mean s/step |
|---|---:|
| score | 0.013325 |
| weight_update | 0.002672 |
| direction | 0.000795 |
| lora_update | 0.000213 |
| build_lora | 0.000000 |

The current checkpoint is not done: sampled vLLM GPU utilization is still about
72% mean in the training window, below the active target. The next profiling
work should focus on direct worker scoring idle time, not on multiple
perturbation directions.

Accepted batch sweep:

| batch | LOZO loss drop | vLLM loss drop | LOZO total s/step | vLLM total s/step | vLLM speedup |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.621094 | 0.624415 | 0.093647 | 0.075339 | 1.2430 |
| 32 | 0.539062 | 0.547690 | 0.115940 | 0.104498 | 1.1095 |
| 64 | 0.519531 | 0.545450 | 0.168678 | 0.165050 | 1.0220 |
| 128 | 0.554688 | 0.565803 | 0.279311 | 0.298030 | 0.9372 |

Detailed vLLM timing:

| batch | score_generate_s | score_generate share | lora_update_s | weight_update_s | build_lora_s | direction_s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.045479 | 61.52% | 0.010884 | 0.007866 | 0.007176 | 0.002326 |
| 64 | 0.135545 | 82.68% | 0.010531 | 0.007933 | 0.006862 | 0.002321 |
| 128 | 0.266845 | 89.90% | 0.010867 | 0.008083 | 0.006954 | 0.002404 |

Earlier batch 16 single-run summary:

| backend | profile | steps | batch | initial | final | loss_drop | loss_drop/s | total_s/step | speedup vs LOZO |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LOZO | minimal | 1000 | 16 | 5.132812 | 4.511719 | 0.621094 | 0.006668 | 0.093145 | 1.00x |
| vLLM | minimal, eager=0 | 1000 | 16 | 5.132618 | 4.508140 | 0.624478 | 0.008687 | 0.071890 | 1.30x |
| vLLM | detailed, eager=0 | 1000 | 16 | 5.132618 | 4.510873 | 0.621745 | 0.008733 | 0.071193 | 1.31x |

`total_s/step` is the primary cross-backend speed metric. LOZO's raw
`step_s_mean` times only the plus/minus low-rank ZO step and excludes trainer
loop/update overhead, so it is not directly comparable to vLLM's raw step
timer.

vLLM detailed q=1 timing:

| component | mean s/step | share of raw step |
|---|---:|---:|
| score | 0.044184 | 62.13% |
| score_generate | 0.044025 | 61.90% |
| score_request_build | 0.000007 | 0.01% |
| score_postprocess | 0.000079 | 0.11% |
| lora_update | 0.010190 | 14.33% |
| weight_update | 0.007780 | 10.94% |
| build_lora | 0.006700 | 9.42% |
| direction | 0.002235 | 3.14% |

Conclusion: q=1 vLLM is faster than the minimal LOZO baseline at batch 16 and
32, essentially tied at batch 64, and slower at batch 128. The scoring
bottleneck is almost entirely `llm.generate(prompt_logprobs=1)` plus temporary
LoRA routing; request construction and NLL postprocess are negligible. During
the live run, LOZO often drove its assigned GPU close to full utilization while
vLLM frequently stayed near the 50% SM-utilization range, even at batch 64 and
128.

## Output Layout

Generated files are written under:

```text
phase3/results/<run_id>/
```

This directory is git-ignored. Logs should include the key parameters and run
timestamp in the filename.

Each non-dry-run launcher also writes:

```text
phase3/results/<run_id>/manifest.json
```

The manifest records the experiment parameters, tmux commands, collect command,
validation command, and whether the launcher had to create the tmux session used
for that run.

Older q=1 JSON files may not include `num_samples` because that option was
added after the first accepted speed result. Validators keep those legacy files
usable and surface the missing field as a warning; new runs should include it in
all result configs.

## Example Commands

Use externally assigned GPUs through `CUDA_VISIBLE_DEVICES`.

The q=1 speed suite can be launched through tmux with externally assigned GPU
IDs:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/preflight_q1.py \
  --tmux-session zo-vllm \
  --batch-size 16 \
  --num-samples 1000
```

`preflight_q1.py` expects numeric GPU indices by default. If a site-specific
CUDA UUID or MIG identifier is required, pass `--allow-non-numeric-gpu` to
preflight, readiness checks, and launchers, then verify the printed warning
manually.

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/launch_q1_speed_suite.py \
  --tmux-session zo-vllm \
  --steps 1000 \
  --batch-size 16 \
  --num-samples 1000 \
  --include-detailed
```

This launches LOZO minimal and vLLM minimal in parallel. With
`--include-detailed`, vLLM detailed profiling runs after vLLM minimal in the
same tmux window, so the same vLLM GPU is not double-booked.

Use `--dry-run` first to print the exact tmux commands without launching jobs.
The launcher prints an `experiment_parameters=...` JSON line before the tmux
commands so the full q=1 configuration is captured in the terminal log. It also
prints `preflight_command=...` and `collect_command=...`; with
`--include-detailed`, it prints `validate_command=...` for the required detailed
timing result.

In non-dry-run mode, launchers run preflight automatically before starting tmux
windows and create the target tmux session if it does not already exist.
Launchers refuse to write into an existing `phase3/results/<run_id>` directory,
even if `--skip-preflight` is used.

To rerun the q=1 batch-size sweep without changing the perturbation direction
count:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/preflight_q1.py \
  --tmux-session zo-vllm \
  --batch-sizes 16,32,64,128 \
  --num-samples 1000
```

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/launch_q1_batch_sweep.py \
  --tmux-session zo-vllm \
  --batch-sizes 16,32,64,128 \
  --num-samples 1000 \
  --steps 1000 \
  --dry-run
```

Remove `--dry-run` after reviewing the printed commands. The launcher runs LOZO
and vLLM in two tmux windows, with each window processing the batch sizes
sequentially on its assigned GPU. It also prints a `collect_command=...` line for
the generated batch directories and a `validate_command=...` line for the
paired result check.

The accepted q=1 batch sweep used:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/launch_q1_batch_sweep.py \
  --tmux-session zo-vllm \
  --run-id phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live \
  --batch-sizes 16,32,64,128 \
  --num-samples 1000 \
  --detailed-batches 16,64,128 \
  --steps 1000
```

Use this local readiness gate before launching a real q=1 batch sweep:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/ready_q1_batch_sweep.py \
  --run-id <RUN_ID> \
  --batch-sizes 16,32,64,128 \
  --num-samples 1000 \
  --steps 1000
```

After all tmux windows finish, collect the generated JSON files into a summary:

```bash
.venv/bin/python -u phase3/runners/collect_q1_batch_sweep.py \
  phase3/results/<run_id> \
  --output phase3/results/<run_id>/summary.md
```

Validate the batch sweep before treating it as a result:

```bash
.venv/bin/python -u phase3/runners/validate_q1_batch_sweep.py \
  phase3/results/<run_id> \
  --expected-batches 16,32,64,128 \
  --require-detailed-batches 16,64,128 \
  --output phase3/results/<run_id>/validation.json
```

Or run the manifest-driven finalizer, which executes the run's recorded collect
and validate commands:

```bash
.venv/bin/python -u phase3/runners/finalize_q1_run.py \
  phase3/results/<run_id>
```

To inspect progress or verify which artifacts have appeared without modifying
the run directory:

```bash
.venv/bin/python -u phase3/runners/status_q1_run.py \
  phase3/results/<run_id>
```

To regression-test the Phase 3 helper tools without launching GPU work:

```bash
.venv/bin/python -u phase3/runners/selftest_q1_tools.py
```

To isolate the q=1 vLLM scoring bottleneck on one externally assigned GPU:

```bash
CUDA_VISIBLE_DEVICES=<GPU> .venv/bin/python -u \
  phase3/runners/microbench_vllm_scoring.py \
  --batch-size 16 \
  --num-samples 1000 \
  --steps 200 \
  --warmup-steps 5 \
  --enforce-eager 0 \
  --modes base,static_lora,rewrite_lora_only,score_with_rewrite \
  --output-dir phase3/results/<run_id>/microbench_b16
```

The microbenchmark keeps `q=1` and reports effective requests per second for:

```text
base                 # prompt_logprobs scoring without LoRA requests
static_lora          # fixed plus/minus LoRA slots, no per-step slot rewrite
rewrite_lora_only    # direction/build/slot rewrite, no generate call
score_with_rewrite   # current q=1 scoring path without optimizer weight update
```

Use it at batch `16`, `64`, and `128` before changing the training algorithm.

To run the full local readiness gate before launching a q=1 batch sweep:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/ready_q1_batch_sweep.py \
  --run-id <run_id> \
  --batch-sizes 16,32,64,128 \
  --detailed-batches 16,64,128 \
  --steps 1000 \
  --num-samples 1000
```

Validate the q=1 speed suite before using it as a formal result:

```bash
.venv/bin/python -u phase3/runners/validate_q1_speed_suite.py \
  phase3/results/<run_id> \
  --min-vllm-speedup 1.2 \
  --output phase3/results/<run_id>/validation.json
```

If minimal and detailed runs are split across directories, pass both
directories and write a combined summary:

```bash
.venv/bin/python -u phase3/runners/collect_q1_speed_suite.py \
  phase3/results/<minimal_run_id> \
  phase3/results/<detailed_run_id> \
  --output phase3/results/<minimal_run_id>/summary_combined.md
```

```bash
CUDA_VISIBLE_DEVICES=<GPU> third_party/LOZO/large_models/.venv/bin/python -u \
  phase3/runners/train_lozo_baseline_perf.py \
  --profile-mode minimal \
  --steps 1000 \
  --batch-size 16 \
  --num-samples 1000 \
  --rank 8 \
  --lr 3e-7 \
  --eps 1e-3 \
  --step-interval 50 \
  --eval-interval 100 \
  --output-dir phase3/results/<run_id>/lozo_minimal
```

```bash
CUDA_VISIBLE_DEVICES=<GPU> VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv/bin/python -u \
  phase3/runners/train_vllm_perf.py \
  --profile-mode minimal \
  --steps 1000 \
  --batch-size 16 \
  --num-samples 1000 \
  --rank 8 \
  --lr 3e-7 \
  --eps 1e-3 \
  --step-interval 50 \
  --eval-interval 100 \
  --enforce-eager 0 \
  --output-dir phase3/results/<run_id>/vllm_minimal_eager0
```

## Acceptance Gates

- Phase 3 scripts do not modify Phase 2 runner behavior.
- Minimal LOZO and minimal vLLM runs produce JSON with config, initial/final
  loss, eval losses, and timing.
- A valid comparison report distinguishes loop time from end-to-end wall-clock
  time and states whether the baseline is minimal or instrumented.
- `validate_q1_speed_suite.py` passes on a fixed-batch speed run directory with
  all required artifacts present and vLLM minimal speedup above the configured
  threshold.
- `validate_q1_batch_sweep.py` passes on a q=1 batch sweep with all expected
  LOZO/vLLM pairs and required detailed batches present.
- Scoring-path optimization claims are backed by
  `microbench_vllm_scoring.py` output, not inferred from total step time alone.
- Results are generated by rerunning scripts, not by hand-moving old logs.
