# Phase 7 Current Conclusion

Date: 2026-06-21

This note records the current Phase 7 position for paper and experiment
planning. It is intentionally conservative: it states what is already supported
by the current implementation and logs, and separates that from the remaining
QoS work.

## One-line Position

Phase 7 has a working read-only-base path: the main weights can stay unchanged,
the base model can be quantized, and ZO steps can be inserted into a serving
load through scheduled compact-NLL requests. The insertion is stable in the
tested setup, but tail latency is still the main unsolved QoS issue.

## Main-weight Invariance

The Phase 7 update mode keeps the base weights read-only.

- `--quantized-update-mode lora_bank` stores high-precision updates in the LoRA
  update bank instead of writing back to base weights.
- Plus and minus probes use two full LoRA bank slots.
- The ZO direction `rank` is independent from `update_bank_rank`, which is the
  capacity of the update bank.
- Bank mode must not call base-weight write-back or fold updates into original
  weights.
- Native checkpoint saving should preserve this semantic: `lora_bank` means
  read-only base weights, not folded weights.

This is the central Phase 7 claim: the trainable state is separated from the
sovereign/base weights.

## Quantization Position

Because the base weights are read-only, the base model can be quantized or
pre-quantized while the update bank remains high precision.

Current paper-safe wording should be modest:

- Quantization reduces base-weight memory pressure.
- FP8 can bring serving-side speedup in the tested stack.
- The Phase 7 design is compatible with pre-quantized base models because ZO
  updates do not require rewriting the quantized base.

Do not make the quantization result the main Phase 7 claim yet. The stronger
claim is read-only base weights plus high-precision update bank.

## Serving-time ZO Insertion

The current serving-time path uses vLLM scheduled compact-NLL scoring rather
than the full direct-worker scoring RPC in the hot serving path.

Current stable strategy:

- Foreground requests use normal serving priority.
- ZO scoring requests use low priority.
- ZO requests are released through a queued admission controller with a token
  budget.
- Slot writes complete before the corresponding ZO scoring requests are
  submitted.
- At most one plus/minus ZO pair is active per bank slot pair before the slot is
  reused.

This path can complete background ZO training while foreground benchmark
requests are running. It is not yet a QoS-optimal scheduler.

## QoS Evidence

Recommended comparison setup:

- Model: `Qwen/Qwen3-8B`
- Workload: random benchmark, input length 256, output length 128
- Request rate: 10 RPS
- Burstiness: 0.3
- Prompts: 600
- GPU memory utilization: 0.82
- ZO: SST-2, rank 8, `update_bank_rank=32`, 100 steps
- Admission: queued token budget
- Scheduler trace enabled

Exact no-training baseline:

`phase7/logs/serving/p7_queued_budget_baseline_rps10_b03_n600_rank8_gpu5_20260621_134705`

| Run | Foreground failed | Throughput ratio | Goodput ratio | p99 TTFT change | p99 TPOT change | ZO steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cap512 | 0 | 0.9999 | 0.9749 | +226.7% | +118.6% | 100 |
| cap1024 | 1 | 0.9983 | 0.9817 | +214.5% | +91.2% | 100 |
| cap2048 | 0 | 1.0000 | 0.9600 | +695.0% | +116.8% | 100 |

Interpretation:

- Request throughput is essentially preserved in these runs.
- Goodput is close but not always within a strict 98% target, depending on the
  chosen cap and whether one accepts the single failed foreground request in
  cap1024.
- Tail latency is still substantially worse. This is the current main weakness.
- cap512 and cap1024 are the useful current data points; cap2048 is too
  aggressive for a headline result.

## Scheduler Trace Evidence

| Run | Total batches | Foreground-only | Mixed | ZO-only | ZO token share |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 4538 | 4536 | 0 | 0 | 0.00% |
| cap512 | 3521 | 3316 | 200 | 1 | 22.46% |
| cap1024 | 3755 | 3591 | 161 | 1 | 22.49% |
| cap2048 | 3779 | 3591 | 153 | 17 | 22.46% |

The ZO requests are not merely queued until after the benchmark. They are mixed
into serving-time scheduler batches and complete during the foreground run.

## Numerical Correctness

The fixed pure-training reference is:

`phase7/logs/serving/references/qwen3_8b_sst2_seed42_rank8_nu50_steps100_bank32.reference.md`

Reference source JSONL:

`phase7/logs/serving/Qwen__Qwen3-8B_p7_ref_train_20260617_155616_rank32/zo_metrics.jsonl`

Reference final metrics:

- Final eval loss: `0.7510829005940393`
- Final eval accuracy: `0.640625`
- Final train `loss_plus`: `0.21501141997846057`
- Final train `loss_minus`: `0.29999794690298076`
- Final projected gradient: `-42.493263462260096`
- Errors: `0`

Correctness comparison against the fixed reference:

| Run | Correctness pass | Errors | Final eval loss diff | Final eval accuracy diff |
| --- | --- | ---: | ---: | ---: |
| cap512 | yes | 0 | +0.0011375839 | 0.0 |
| cap1024 | yes | 0 | -0.0005293618 | 0.0 |
| cap2048 | no | 0 | -0.0004886756 | -0.00390625 |

The cap512 and cap1024 serving-training runs pass the current correctness
acceptance check: final eval loss is within `0.005`, final eval accuracy matches
exactly, all 100 train steps are present, and no runtime errors were recorded.

This does not claim bitwise identity. Step-level plus/minus losses and projected
gradients are not exactly identical to the pure-training reference, which is
expected for scheduled serving execution and floating-point/batching differences.
The current evidence supports "no detected algorithmic miscompute" for cap512
and cap1024, not "bit-exact replay".

## Paper-safe Summary

Suggested wording:

> Phase 7 keeps the base model weights immutable and stores ZO updates in a
> high-precision LoRA update bank. This makes the base model compatible with
> quantized or pre-quantized deployment while still allowing training-time
> perturbation and update accumulation. In serving experiments, background ZO
> steps can be inserted through low-priority scheduled compact-NLL requests and
> complete stably under foreground load. In the current implementation,
> throughput is largely preserved, while tail latency remains the primary QoS
> cost and motivates the next scheduler/admission-control iteration.

## What Not To Overclaim

- Do not claim p99 TTFT/TPOT is solved.
- Do not use cap2048 as the strict correctness-pass headline.
- Do not claim bit-exact equality with the pure-training reference.
- Do not make FP8 speedup the main result yet; mention it only as a compatible
  benefit of read-only quantized base weights.
