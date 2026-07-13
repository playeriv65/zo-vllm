# Phase7

Phase7 studies read-only base weights with high-precision update banks.

## Part 1: Read-Only Base Weight

Part 1 keeps quantized or full-precision base weights read-only and routes all
ZO updates through the LoRA bank path. The update rank used for the ZO direction
is independent from the larger bank rank used by vLLM LoRA slots.

## Part 2: Serving Load With Background ZO

Part 2 runs a normal text vLLM server while a low-priority background ZO trainer
shares the same engine. Foreground serving requests use the regular OpenAI
endpoint and priority `0`; background scoring requests are submitted through the
vLLM scheduler with priority `1000`.

By default, the trainer delegates direction sampling and LoRA-bank update state
to `zo_vllm.training.worker_update_bank`, which keeps them resident inside the
vLLM worker as an update backend. The server process sends only small control
payloads for slot prepare and scalar update apply. This avoids sending hundreds
of CUDA tensors through `collective_rpc` every step. Scheduled compact NLL
forward passes keep priority `1000`.

The Phase 7 runner uses HTTP only for start, stop, and status. The API server
stores a passive `BackgroundTrainingHandle` containing the current HF Trainer,
thread, stop signal, and observations; free API functions own the lifecycle.
The shared Hugging Face `ZOTrainer` runs in one dedicated OS thread inside the
API-server process. Dataset shuffle/map, sampler/DataLoader, ragged collation,
task loss, metrics, callbacks, optimizer ordering, scheduler, and eval cadence
are the same native HF path as offline training. There is no serving trainer or
serving controller. `ScheduledServingRuntime` is the narrow adapter that blocks
the trainer thread on `BlockingAsyncBridge`; `AsyncZOEngineService` owns QoS
admission, the plus/minus pair lock, scheduled option-NLL scoring, and worker
RPC. No async optimizer or async trainer lifecycle is required.

Slot writes do not wait for foreground HTTP requests to drain. They only obey
the ZO pair barrier: write plus/minus slots, wait for an idle foreground-load
gap before submitting the matching low-priority scoring requests, wait for both
to finish, and only then allow the next slot write. Foreground serving requests
do not use the serving ZO plus/minus LoRA IDs, so slot writes remain independent
from foreground execution, while score admission stays conservative.

`SLOT_WRITE_STREAM=background_sync` is the conservative default copy path.
Serving ZO slot writes are enqueued on a separate CUDA stream, then `prepare_slots`
waits for the copy event before plus/minus compact-NLL requests are submitted to
the scheduler. This keeps request admission out of the runner-side event-fence
path while preserving the pair barrier.

Serving-time ZO has four score admission policies. The conservative default
`idle_gap` waits only before score submission until the server foreground-load
tracker reports an empty gap. The more aggressive `scheduler_only` policy does
not wait for a foreground gap: after the plus/minus slots are written, it submits
the low-priority compact-NLL score pair immediately and lets the vLLM scheduler
order the work. The `queued` policy also skips the foreground-gap wait, but puts
all ZO compact-NLL requests into a FIFO submit queue first; a token bucket and
max-inflight guard then release requests to vLLM gradually. It also keeps an
admitted-token cap: a request's estimated prompt tokens count against the cap
after it is released to vLLM and are freed only after the compact-NLL result
returns. This keeps training and eval semantics unchanged while smoothing
request admission. The vLLM scheduler itself remains unmodified and uses normal
priority scheduling. The `gpu_utilization` policy waits until `nvidia-smi`
reports the target GPU at or below a configured utilization threshold before
submitting the next ZO score pair. The `idle_gap` and `gpu_utilization` waits
have no timeout by default and exit cleanly when the trainer is stopped.

In compile/CUDA graph mode, serving ZO compact-NLL LoRA scoring also defaults to a
per-batch eager fallback through `VLLM_ZO_FORCE_EAGER_SCORING=1`. This
skips the compiled model wrapper and CUDA graph only for batches that actually
contain serving ZO LoRA compact-NLL scoring; foreground-only serving batches keep
the server's normal execution mode.

The private trainer router is only enabled when:

```bash
VLLM_ZO_SERVING_TRAINING=1
```

Endpoints:

- `POST /zo_vllm/serving_zo/start`
- `POST /zo_vllm/serving_zo/stop`
- `GET /zo_vllm/serving_zo/status`

The formal serving path does not use full direct-worker scoring RPCs. Worker RPCs
are used only for LoRA slot initialization, worker update-bank slot prepare,
and scalar update apply.

The trainer validates the server configuration at `/start`. It requires
priority scheduling, LoRA enabled, at least two LoRA slots, and
`max_lora_rank >= update_bank_rank`.

## Timeline Verification

Use the timeline analyzer after a serving run to check whether ZO actually
interleaved with foreground requests:

```bash
.venv/bin/python phase7/scripts/analyze_serving_timeline.py \
  --bench-json phase7/logs/serving/<run>/baseline_rps2.json \
  --zo-jsonl phase7/logs/serving/Qwen__Qwen3-8B_<run>/zo_metrics.jsonl \
  --summary-json phase7/logs/serving/<run>/timeline_summary.json
```

Key fields:

- `steps_overlap_bench`: ZO steps whose wall-clock interval overlaps the bench
  window.
- `steps_completed_before_bench_end`: ZO steps that finished before the
  foreground benchmark window ended. This rules out post-load backfill.
- `score_intervals_overlapping_requests`: scheduled scoring intervals that
  overlap foreground request intervals.
- `slot_write_intervals_overlapping_requests`: LoRA slot prepare intervals that
  overlap foreground request intervals.
- `bench_step_density_per_s`: ZO step density inside the foreground benchmark
  window.
- `score_submits_with_foreground_load`: scoring submissions that saw positive
  foreground load. In the conservative idle-gap path this should be zero or
  very close to zero.
- `slot_writes_with_foreground_load`: slot writes performed while foreground
  load was positive. This is allowed by the default nonblocking slot-write path.
- `mean_score_admission_wait_s` / `p99_score_admission_wait_s`: how long ZO
  waited for a foreground gap before submitting score requests.
- `steps_after_bench`: ZO steps that only ran after foreground load finished.

Historical GPU6 Qwen3-8B / random text / RPS=10 probes found nonfinite compact
NLL failures when foreground no-LoRA requests were co-batched with serving ZO
LoRA scoring. The root cause was missing direct-LoRA metadata and shape
validation for active modules; the current path validates the generated LoRA
scope against the registered vLLM LoRA modules before worker-bank use.

The default serving insertion setting is:

```bash
INTER_STEP_DELAY_S=0.0
SLOT_WRITE_STREAM=background_sync
ZO_ADMISSION_POLICY=idle_gap
VLLM_ZO_FORCE_EAGER_SCORING=1
```

On a controlled Qwen3-8B / random text / RPS=2 smoke with 30 prompts,
`INPUT_LEN=256`, `OUTPUT_LEN=128`, `RANK=2`, and `UPDATE_BANK_RANK=16`, this
kept throughput and goodput at 99.87% of the no-training baseline while 11
scheduled scoring intervals and 11 slot-write intervals overlapped foreground
requests. The strict smoke acceptance passed with p99 TTFT/TPOT/E2E deltas
within 5%.

## Scheduler Trace and QoS Attribution

Set `SCHED_TRACE=1` to make the serving runner pass
`VLLM_ZO_SCHED_TRACE_PATH=<log_root>/scheduler_trace.jsonl` to the vLLM worker.
The trace is opt-in and records one JSONL row per GPU scheduler batch:
scheduled request ids, priority, LoRA id/name, scheduled token count,
prefill/decode token split, rough attention key-token work, queue sizes,
preemptions, and finished requests.

Offline analysis helpers:

```bash
.venv/bin/python phase7/scripts/analyze_scheduler_trace.py \
  --trace phase7/logs/serving/<run>/scheduler_trace.jsonl \
  --out-json phase7/logs/serving/<run>/scheduler_trace_summary.json \
  --out-md phase7/logs/serving/<run>/scheduler_trace_summary.md \
  --max-target-ms 1000

.venv/bin/python phase7/scripts/compare_serving_qos.py \
  --base phase7/logs/serving/<baseline>/baseline_rps10.json \
  --candidate phase7/logs/serving/<zo-run>/zo_on_rps10.json \
  --out-json phase7/logs/serving/<zo-run>/scheduler_only_vs_baseline.qos.json \
  --out-md phase7/logs/serving/<zo-run>/scheduler_only_vs_baseline.qos.md

.venv/bin/python phase7/scripts/analyze_request_qos_trace.py \
  --base phase7/logs/serving/<baseline>/baseline_rps10.json \
  --candidate phase7/logs/serving/<zo-run>/zo_on_rps10.json \
  --trace phase7/logs/serving/<zo-run>/scheduler_trace.jsonl \
  --out-json phase7/logs/serving/<zo-run>/request_qos_trace_impact.json \
  --out-md phase7/logs/serving/<zo-run>/request_qos_trace_impact.md

.venv/bin/python phase7/scripts/compare_serving_zo_correctness.py \
  --reference phase7/logs/serving/references/<reference>.jsonl \
  --candidate phase7/logs/serving/<zo-metrics>/zo_metrics.jsonl \
  --out-json phase7/logs/serving/<zo-run>/serving_zo_correctness.json \
  --out-md phase7/logs/serving/<zo-run>/serving_zo_correctness.md
```

Recent Qwen3-8B / random text / RPS=10 / `BURSTINESS=0.3` traces show that vLLM
priority ordering works as expected: foreground requests with priority `0` are
scheduled before ZO requests with priority `1000`. QoS still degrades when the
same GPU batch also includes a large amount of ZO prefill/eval tokens, because
the whole batch becomes heavier even though the foreground decode tokens appear
first in request order.

The scheduler-trace analyzer also fits a small linear model using the interval
to the next scheduler batch as a proxy for batch cost. Features are foreground
and ZO prefill/decode token counts plus rough prefill/decode attention
key-token totals. This is intentionally a diagnostic model, not a profiler:
large idle gaps are filtered by `--max-target-ms`, and the output is meant to
identify which batch composition is hurting QoS enough to inspect next.

## Script

The serving-load experiment logic lives in
`zo_vllm.experiment.runners.serving_load`. The `phase7/scripts/` entrypoint is a
thin compatibility wrapper, so existing commands keep working while the runner
is shared from the common experiment package.

Run long-lived modes in tmux:

```bash
GPU=5 RUN_TAG=phase7_serving_server phase7/scripts/run_serving_load_training.sh server
GPU=5 RUN_TAG=phase7_serving_baseline REQUEST_RATE=2 NUM_PROMPTS=1000 \
  phase7/scripts/run_serving_load_training.sh bench
GPU=5 RUN_TAG=phase7_serving_zo REQUEST_RATE=2 NUM_PROMPTS=1000 TRAIN_STEPS=100 \
  phase7/scripts/run_serving_load_training.sh zo-bench
RUN_TAG=phase7_serving_zo REQUEST_RATE=2 \
  phase7/scripts/run_serving_load_training.sh analyze
RUN_TAG=phase7_summary \
  phase7/scripts/run_serving_load_training.sh summarize
BASELINE_JSON=phase7/logs/serving/<baseline>/baseline_rps2.json \
  ZO_SUMMARY_JSON=phase7/logs/serving/<zo-run>/timeline_summary.json \
  phase7/scripts/run_serving_load_training.sh acceptance
```

`zo-bench` starts the background trainer, runs `vllm bench serve`, stops the
trainer by default, writes `timeline_summary.json`, and runs acceptance when
`BASELINE_JSON` is provided.

Useful environment variables:

- `MODEL`: defaults to `Qwen/Qwen3-8B`
- `DTYPE`: vLLM server dtype, defaults to `auto`
- `GPU`: required for `server`; set it explicitly, for example `GPU=5`
- `VLLM_QUANTIZATION`: passed to the vLLM server when set
- `UPDATE_BANK_RANK`: vLLM LoRA bank rank. Defaults to `auto`, which estimates
  capacity from `TRAIN_STEPS`, `NU`, and `RANK` with one extra V-refresh block.
  Auto-estimated and manual values are rounded up to the nearest
  vLLM-supported `max_lora_rank` bucket. Set a positive integer to override it
  manually. Open-ended runs must pass an explicit rank.
- `VLLM_ZO_RESERVED_LORA_BANK_BYTES`: optional manual override for the extra
  worker-resident ZO update-bank memory that vLLM should subtract from KV cache
  capacity. If unset, the serving runner estimates it from the resolved bank
  rank, target modules, model config, and direction dtype.
- `RANK`: ZO direction rank, defaults to `8`
- `SEED`, `WEIGHT_DECAY`, `DIRECTION_DTYPE`: background ZO numeric settings;
  `DIRECTION_DTYPE=auto` follows the model config, with FP32 configs treated as
  FP16 to match vLLM's default generation dtype behavior
- `DATASET_NAME`, `DATASET_PATH`, `HF_SPLIT`: foreground bench dataset settings;
  `DATASET_NAME=random` leaves `DATASET_PATH` empty by default
- `INPUT_LEN`, `OUTPUT_LEN`, `TEMPERATURE`: optional controlled bench settings;
  useful with `DATASET_NAME=random`
- `REQUEST_RATE`, `BURSTINESS`, `BENCH_SEED`, `NUM_PROMPTS`: foreground
  `vllm bench serve` load-shape settings. Use `BURSTINESS < 1` to create
  burstier traffic with local saturated windows and idle gaps.
- `SCHED_TRACE`: set to `1` to write `scheduler_trace.jsonl` for GPU-batch
  attribution. When the trace exists, `analyze` also writes
  `scheduler_trace_summary.json` / `.md` with token-feature totals and the
  simple batch-interval cost model.
- `TRAIN_STEPS`, `BATCH_SIZE`, `EVAL_INTERVAL`, `INITIAL_EVAL`: background
  ZO trainer settings
- `LR_SCHEDULER_TYPE` and `WARMUP_STEPS` configure the Transformers scheduler.
  Existing scripts omit them and retain the historical constant-LR behavior.
- `GRADIENT_ACCUMULATION_UPDATE_STEPS`, `U_BETA`, and `U_NORM_CAP` configure the
  worker-resident update bank.
- Background training uses Hugging Face `Dataset.shuffle` and the native
  Trainer sampler. Phase 7 no longer exposes the legacy NumPy row shuffle.
- `ZO_RANDOM_DEVICE`, `DIRECTION_DEVICE`, `DIRECTION_DTYPE`,
  `DIRECTION_SAMPLING`, `DIRECTION_SCALE`, `PERTURBATION_NORMALIZATION`, and
  `V_NORMALIZATION` configure direction generation. Historical LOZO references
  with an implicit `1 / sqrt(rank)` direction scale map to the current
  `DIRECTION_SCALE=1` and `PERTURBATION_NORMALIZATION=rms` defaults.
- `ZO_TARGET_MODULES`: optional vLLM-style LoRA target module list. Comma or
  whitespace separated values are accepted. If unset, the default transformer
  linear target list is used.
- `ZO_INCLUDE_LM_HEAD`, `ZO_INCLUDE_EMBEDDINGS`: optional outer-level
  convenience switches. When set to `1`, they append `lm_head` or `embed_tokens`
  to the same resolved `target_modules` list used by vLLM, metadata generation,
  direct slot registration, and worker-bank validation.
- `INTER_STEP_DELAY_S`: optional throttle between ZO steps
- `SLOT_WRITE_STREAM`: `background_sync` by default; set to `background` to
  reproduce the runner-side event-fence async path, or `default_sync` to enqueue
  on the normal CUDA stream and synchronize before request submission
- `ZO_ADMISSION_POLICY`: `idle_gap` by default. Set to `scheduler_only` to skip
  the foreground-gap wait and submit ZO scoring directly to the priority
  scheduler. Aliases `none`, `no_gate`, `no_idle_gap`, `immediate`, and
  `scheduler` are accepted and recorded as `scheduler_only`. Set to `queued`
  to put ZO compact-NLL requests into the local submit queue before releasing
  them to vLLM; aliases `queue`, `rate_limited`, and `token_bucket` are accepted.
  Set to `gpu_utilization` to submit only when target GPU utilization is at or
  below the configured threshold; aliases `gpu`, `gpu_idle`, `gpu_util`,
  `util`, and `utilization` are accepted.
- `ZO_ADMISSION_MAX_GPU_UTILIZATION`, `ZO_ADMISSION_GPU_DEVICE`: controls for
  `ZO_ADMISSION_POLICY=gpu_utilization`. The default threshold is `70`, and the
  default device is the configured serving `GPU`.
- `ZO_QUEUE_TOKEN_RATE`, `ZO_QUEUE_BURST_TOKENS`, `ZO_QUEUE_MAX_INFLIGHT`,
  `ZO_QUEUE_MAX_ADMITTED_TOKENS`, `ZO_QUEUE_POLL_S`: local queued-admission
  controls used only when `ZO_ADMISSION_POLICY=queued`. The queue limits submit
  rate and the number of ZO prompt tokens already released to vLLM but not yet
  completed; it does not change training semantics. Upper layers can enqueue
  full plus/minus/eval scoring work and wait for the same aggregate result.
- `ZO_ADMISSION_TIMEOUT_S`: defaults to `0`, meaning no timeout; only applies to
  `idle_gap`, and the wait is still stop-aware.
- `VLLM_ZO_FORCE_EAGER_SCORING`: defaults to `1`; disables CUDA graph
  and skips the compiled model wrapper only for batches containing serving ZO LoRA
  compact-NLL scoring. Set to `0` only to debug or reproduce dynamic LoRA bank
  graph behavior.
- `ENABLE_WANDB`: defaults to `1`; set to `0` for local smoke tests
- `SUMMARY_GLOB`, `BASELINE_SUMMARY`: optional inputs for `summarize` mode
- `BASELINE_JSON`, `ZO_SUMMARY_JSON`: inputs for `acceptance` mode
- `MIN_THROUGHPUT_RATIO`, `MIN_GOODPUT_RATIO`,
  `MAX_P99_LATENCY_DELTA_PCT`, `MIN_OVERLAP_STEPS`: acceptance thresholds
- `STOP_ZO_AFTER_BENCH`: defaults to `1` for `zo-bench`
- `ANALYZE_AFTER_BENCH`: defaults to `1` for `zo-bench`
- `ACCEPT_AFTER_BENCH`: defaults to `1` when `BASELINE_JSON` is set

All logs and benchmark artifacts go under `phase7/logs/serving/`.
