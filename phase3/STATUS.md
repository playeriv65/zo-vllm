# Phase 3 Status

## Latest Checkpoint

Phase 3 is not finished. The current checkpoint only addresses the obvious
q=1 performance blockers found after the initial batch sweep:

- avoid per-step CPU LoRA tensor construction when direct GPU slots are used
- write plus/minus LoRA slots directly from U/V directions inside vLLM
- score prompt token IDs through a direct vLLM worker path instead of the full
  `generate(prompt_logprobs=1)` serving loop
- run warmup steps before timing so one-time Triton/LoRA JIT does not pollute
  measured step time
- time LOZO with a direct wall-clock step metric that includes `lowrank_zo_update`
  and optimizer update overhead
- keep `q=1`; multi-direction batching is still out of scope for this checkpoint

Latest q=1 OPT-1.3B checkpoint:

```text
vLLM: local ignored run ending in 20260524_123159
LOZO: local ignored run ending in 20260524_121646
```

New OPT-1.3B runs should use the `opt1p3b` slug.

Configuration:

```text
model=facebook/opt-1.3b
batch_size=16
num_samples=1024
rank=8
lr=3e-7
eps=1e-3
q=1
scoring_backend=direct_worker
direct_lora_from_directions=1
base_eval_mode=skip
enforce_eager=0
direction_sampling=flat
qkv_weight_update=batched
sync_weight_update=0
```

Latest speed result, using `total_s / measured_steps` as the cross-backend
metric:

| backend | total s/step | notes |
|---|---:|---|
| LOZO minimal | 0.051668 | includes ZO step and update |
| vLLM detailed | 0.017034 | direct worker score, direct slot/update path |
| vLLM speedup | 3.03x | relative to corrected LOZO wall-clock step |

Latest vLLM detailed step breakdown:

| component | mean s/step |
|---|---:|
| score | 0.013325 |
| weight_update | 0.002672 |
| direction | 0.000795 |
| lora_update | 0.000213 |
| build_lora | 0.000000 |

The same run's sampled vLLM GPU utilization is about 72% mean in the training
window, so the active utilization target is not yet met. The remaining hotspot
is direct worker scoring, especially model forward and end-of-score
synchronization. Failed or inconclusive branches are not accepted as results:
fused LoRA+score was marginal, slot pipelining was slower, fused score+weight
update was slower, and caching mutable direct-score attention state across
non-adjacent batches caused a CUDA illegal memory access and was reverted.

This is a speed checkpoint, not a convergence claim. vLLM base loss evaluation
was skipped during the speed runs to keep the benchmark focused on training-step
throughput. Use Phase 2 results for accepted convergence evidence.

The previous q=1 batch sweep below remains useful as the baseline that exposed
the original under-utilization problem.

## Current Scope

Phase 3 is the performance-optimization and testing track. Phase 2 remains the
correctness and convergence baseline and should not be edited for Phase 3 speed
experiments.

The active Phase 3 question is still single-direction ZO training:

```text
q=1
```

Multiple perturbation directions remain out of scope until the q=1 vLLM scoring
bottleneck is isolated. The q=1 LOZO-vs-vLLM speed curve has been measured and
shows that larger training batches alone do not keep vLLM ahead of LOZO.

## Accepted q=1 Batch Sweep

Formal run directory:

```text
phase3/results/phase3_q1_batch_sweep_b16-32-64-128_s1000_20260523_live/
```

Validation:

```text
ok=true
errors=[]
warnings=[]
```

Configuration:

```text
q=1
steps=1000
batch_sizes=16,32,64,128
detailed_batches=16,64,128
num_samples=1000
rank=8
lr=3e-7
eps=1e-3
step_interval=50
eval_interval=0
seed=42
zo_random_device=cuda
train_scope=lora_only
```

vLLM configuration:

```text
batch_invariant=0
enforce_eager=0
lora_residency=gpu
lora_injection=direct
weight_update=direct
weight_update_precision=param
```

Accepted speed result:

| batch | LOZO total s/step | vLLM total s/step | vLLM speedup | LOZO loss drop | vLLM loss drop |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.093647 | 0.075339 | 1.2430 | 0.621094 | 0.624415 |
| 32 | 0.115940 | 0.104498 | 1.1095 | 0.539062 | 0.547690 |
| 64 | 0.168678 | 0.165050 | 1.0220 | 0.519531 | 0.545450 |
| 128 | 0.279311 | 0.298030 | 0.9372 | 0.554688 | 0.565803 |

Detailed vLLM scoring share:

| batch | score_generate_s | score_generate_share |
|---:|---:|---:|
| 16 | 0.045479 | 61.52% |
| 64 | 0.135545 | 82.68% |
| 128 | 0.266845 | 89.90% |

Conclusion: increasing the `q=1` training batch does not make vLLM increasingly
faster. The current path is useful at batch 16 and 32, essentially tied at batch
64, and slower at batch 128. The bottleneck is the vLLM
`generate(prompt_logprobs=1)` scoring path with temporary LoRA routing; NLL
postprocess and request construction are negligible.

During the live run, LOZO often drove the assigned GPU close to full utilization
while vLLM frequently stayed near the 50% SM-utilization range, including at
batch 64 and 128. This is a real performance signal, not just a logging artifact.

## Earlier q=1 b16 Baseline

Formal run directory:

```text
phase3/results/phase3_q1_speed_noeval_b16_s1000_20260523_rerun/
```

Configuration:

```text
steps=1000
batch_size=16
num_samples=1000
rank=8
lr=3e-7
eps=1e-3
step_interval=50
eval_interval=0
seed=42
zo_random_device=cuda
train_scope=lora_only
```

vLLM configuration:

```text
batch_invariant=0
enforce_eager=0
lora_residency=gpu
lora_injection=direct
weight_update=direct
weight_update_precision=param
```

Accepted result:

| backend | total s/step | loss drop | speedup vs LOZO |
|---|---:|---:|---:|
| LOZO minimal | 0.093145 | 0.621094 | 1.0000 |
| vLLM minimal | 0.071890 | 0.624478 | 1.2957 |
| vLLM detailed | 0.071193 | 0.621745 | 1.3083 |

The current bottleneck is vLLM scoring:

```text
score_generate_s=0.044025
score_generate_share=61.90%
```

## Next Required Experiment

Do not move to multiple perturbation directions yet. The next work should
optimize or isolate the vLLM scoring path for `q=1`, because the batch sweep
shows that larger batch alone does not saturate the current implementation.

The next diagnostic script is:

```text
phase3/runners/microbench_vllm_scoring.py
```

It separates:

```text
base                 # prompt_logprobs scoring without LoRA requests
static_lora          # fixed plus/minus LoRA slots, no per-step slot rewrite
rewrite_lora_only    # direction/build/slot rewrite, no generate call
score_with_rewrite   # current q=1 scoring path without optimizer weight update
```

Use a single externally assigned GPU:

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

Repeat at the batch sizes that exposed the utilization problem, especially
`16`, `64`, and `128`. Treat the output JSON as diagnostic evidence, not as a
replacement for the formal LOZO-vs-vLLM training comparison.

Use the same externally assigned GPU discipline. For a full q=1 batch-size
sweep rerun:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/preflight_q1.py \
  --tmux-session zo-vllm \
  --batch-sizes 16,32,64,128 \
  --num-samples 1000
```

Preflight expects numeric GPU indices unless `--allow-non-numeric-gpu` is used
for a deliberate CUDA UUID/MIG value. Pass the same flag to readiness checks and
launchers so the manifest records that choice.

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/launch_q1_batch_sweep.py \
  --tmux-session zo-vllm \
  --batch-sizes 16,32,64,128 \
  --num-samples 1000 \
  --detailed-batches 16,64,128 \
  --steps 1000
```

The launcher prints the exact collection and validation commands. After the
tmux windows finish, run those printed commands before interpreting the result.

## Validation Gates

Before using a Phase 3 run as evidence:

```bash
.venv/bin/python -m py_compile phase3/runners/*.py
git diff --check
.venv/bin/python -u phase3/runners/collect_q1_batch_sweep.py \
  phase3/results/<run_id> \
  --output phase3/results/<run_id>/summary.md
.venv/bin/python -u phase3/runners/validate_q1_batch_sweep.py \
  phase3/results/<run_id> \
  --expected-batches 16,32,64,128 \
  --require-detailed-batches 16,64,128 \
  --output phase3/results/<run_id>/validation.json
```

The validator must report:

```text
ok=true
```

For runs launched after manifest support was added, the preferred collection
path is:

```bash
.venv/bin/python -u phase3/runners/finalize_q1_run.py phase3/results/<run_id>
```

Use this read-only status command while a run is active or after it finishes:

```bash
.venv/bin/python -u phase3/runners/status_q1_run.py phase3/results/<run_id>
```

Use this helper-tool regression test after editing collectors, validators,
finalizers, or status scripts:

```bash
.venv/bin/python -u phase3/runners/selftest_q1_tools.py
```

Before launching the q=1 batch sweep, run:

```bash
PHASE3_LOZO_GPU=<GPU> PHASE3_VLLM_GPU=<GPU> .venv/bin/python -u \
  phase3/runners/ready_q1_batch_sweep.py \
  --run-id <run_id> \
  --batch-sizes 16,32,64,128 \
  --detailed-batches 16,64,128 \
  --steps 1000 \
  --num-samples 1000
```

Older Phase 3 q=1 JSON files were generated before `num_samples` was added to
the runner config. Validators keep those results usable and report the missing
field under `warnings`; new runs should include `num_samples` in every LOZO and
vLLM config.

## Runtime Rules

- Do not hard-code GPU IDs in code or docs.
- Use `PHASE3_LOZO_GPU` and `PHASE3_VLLM_GPU`, or pass `--lozo-gpu` and
  `--vllm-gpu`.
- Keep generated outputs under `phase3/results/`.
- Keep `phase3/results/` git-ignored.
- Do not commit result JSON or logs.
- Treat `phase3/results/<run_id>/manifest.json` as the run-local source of
  truth for parameters and collection/validation commands.
- Launchers run preflight automatically in non-dry-run mode and store the
  preflight command/report in `manifest.json`.
- Launchers refuse to write into an existing run directory, even with
  `--skip-preflight`.
- Launchers create the target tmux session in non-dry-run mode if it is missing.
