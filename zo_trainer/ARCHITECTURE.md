# ZO Trainer Architecture

## Purpose

`zo_trainer` is the Hugging Face `Trainer` integration layer for ZO-vLLM. It is
not a wrapper around the older HF-like `VLLMZOTrainer`, and it is not a
task-adapter framework. Its job is to let Hugging Face own the standard training
surface while ZO-vLLM owns only the runtime behavior that cannot be expressed as
normal autograd.

## Ownership Boundary

```text
Hugging Face ecosystem
  load_dataset / Dataset.map / tokenizer or processor / chat template
  DataCollator extension point / TrainingArguments / callbacks / metrics / logging
    -> zo_trainer.ZOTrainer
       -> ZOTrainerModel.forward(**hf_batch) for clean logits
       -> ZOTrainer.compute_loss_from_outputs(...) for HF loss
       -> zo_model.estimate_with_score_fn(...) for estimator-driven probing
          -> ZOStepper
             -> DirectionProvider
             -> ZOEstimator
                -> engine-style scoring calls
                -> ZOGradientEstimate
             -> ZOPendingStep
       -> ZOSGDOptimizer.step()
          -> UpdateState
          -> ZOVLLMEngine / vLLM direct scoring / LoRA runtime
```

Hugging Face owns every step that is not ZO-specific:

- dataset loading and split selection;
- `Dataset.map(...)` preprocessing;
- tokenizer, processor, and chat template application;
- `input_ids`, `attention_mask`, `labels`, and `labels=-100` loss-mask schema;
- sampling, batching, and DataLoader orchestration through Hugging Face;
- evaluation and prediction orchestration;
- clean and perturbed loss computation through `compute_loss_func` or the
  corresponding Transformers loss utility;
- `compute_metrics`, callbacks, reporting, and checkpoint cadence.

ZO-vLLM owns only:

- vLLM-backed scoring or logits production;
- plus/minus perturbation slots;
- direction sampling and estimator semantics;
- projected-gradient update application;
- runtime-aware checkpoint payloads.

## Data Path

The primary data path is the ordinary Hugging Face path:

```python
raw = load_dataset(...)
tokenized = raw.map(preprocess_function, batched=True, remove_columns=...)
collator = VLLMDataCollator()  # native HF data_collator extension point
trainer = ZOTrainer(
    model=...,
    train_dataset=tokenized,
    data_collator=collator,
    processing_class=tokenizer,  # native HF save/load ownership
)
```

The batch passed into `ZOTrainer` must be a normal Hugging Face batch such as:

```python
{
    "input_ids": list[list[int]],  # ragged, no sequence padding
    "labels": list[list[int]],     # ragged; ignored positions are -100
}
```

`VLLMDataCollator` is the default because dense PyTorch collators would create a
CPU `pad -> tensor -> list -> unpad` round trip that vLLM does not need. It is a
normal Hugging Face collator callable, so Dataset, sampler, DataLoader, Trainer,
callbacks, and metrics remain native HF facilities. It also flattens prompt
classification options while preserving `option_loss_token_counts`,
`row_option_counts`, and row labels. No tokenizer call occurs in the collator or
training hot path.

The Trainer and direct model paths require ragged Python sequence batches and
leave them on CPU. Dense tensor batches are rejected. vLLM may pad the flattened
total token count internally for CUDA
graph or kernel shapes; that is a backend execution detail, not per-sequence
padding from the HF batch.

`ZOTrainerArguments.use_cpu` is always true: Hugging Face owns CPU-side
orchestration while vLLM owns every GPU device. The adapter does not rewrite
Trainer's private `_n_gpu` or `_train_batch_size` fields.

`zo_trainer.preprocessing` provides small `Dataset.map` helpers for common LM
cases, but those helpers emit only standard Hugging Face fields. They do not
construct `TokenProbeBatch`, do not compute objective losses, and do not encode
dataset-specific trainer logic.

## Model Boundary

`ZOTrainerModel.forward(**batch)` is the runtime model facade. It accepts native
Hugging Face tokenized inputs and returns compact logits plus aligned loss
labels. It never computes task loss. For a clean eval call it scores the current
model; for a ZO train probe the runtime selects the internal LoRA slot(s).
Trainer code does not choose or manufacture LoRA IDs.

For causal LM objectives, the runtime returns compact vocabulary logits only for
active label positions. Labels with `-100` are not materialized. For prompt
classification, it returns per-option request-NLL tensors, which are the model
outputs needed to construct class logits. Under the single-process vLLM executor
both output forms remain on GPU. `ZOTrainer` computes the canonical causal or
classification loss from those outputs, using public `compute_loss_func` when
supplied and Transformers loss utilities otherwise.

The worker does not compute token NLL again when causal logits are requested.
Classification request NLL is computed once on GPU and is not copied to CPU.
After HF computes every probe-group loss, all group scalars are transferred to
CPU together once because the ZO estimator and update coefficient are scalar
control values. Multi-process executors use the explicit CPU fallback because
CUDA tensors cannot cross that RPC boundary safely.

## Estimator Boundary

The estimator layer is the algorithm boundary. `ZOTrainer` and
`ZOTrainerModel` must not hard-code plus/minus, one-sided, multi-query, or
evolution-strategy behavior. They only expose Hugging Face batch-to-loss
semantics. `ZOStepper` calls the configured `ZOEstimator`, and the estimator
decides which probes are needed.

Current estimator ownership:

| Layer | Owns | Must Not Own |
| --- | --- | --- |
| `ZOTrainer` | HF loop integration, LR scheduler, estimate staging, post-forward loss, logging/eval/save cadence | LoRA slot IDs, perturbation sides, number of queries |
| `ZOTrainerModel` | One-time HF batch conversion and compact-logits scoring | Task loss, direction sampling, estimator branching, slot allocation policy |
| `ZOStepper` | callback ordering, direction sampling, estimate construction, runtime apply orchestration | HF scheduler/optimizer policy, dataset/tokenizer semantics |
| `ZOEstimator` | antithetic, one-sided, multi-query, ES probe plan and aggregation | HF dataloading or metric APIs |
| `ZOSGDOptimizer` | HF optimizer protocol, LR/weight decay, step ordering, one staged estimate per step | Probe planning, LoRA slots, runtime tensor execution |
| `ZOUpdateState` | Efficient vLLM weight mutation, low-rank accumulation, and fold behavior | LR scheduling or estimator policy |
| `zo_vllm.core.probe_results` | complete objective-group losses computed by HF | Token-NLL emulation or estimator policy |
| `zo_vllm.core.token_scores` | real request-level token-score chunking, merging, and slicing | HF objective losses or estimator policy |
| `ZOVLLMEngine` | LoRA slot writes, direct scoring/logits, runtime checkpoint payloads | Trainer cadence or task-specific preprocessing |

`estimate_with_score_fn(...)` preserves this boundary by passing a minimal
`TokenGroupScorer` beside the real runtime engine. It does not construct a
second engine facade. Estimators use the scorer for objective values and the
engine for runtime operations:

- `score_plus_minus_directions(...)` for antithetic two-sided probes;
- `score_token_groups(...)` for clean or one-sided probe scoring;
- `set_probe_directions(...)` for opaque runtime slot assignment;
- `score_probe_slots_with_scorer(...)` or `generate_probe_slots(...)` without
  exposing physical IDs.

The adapter passes runtime token groups directly to the scorer. It does not
reconstruct an HF tensor batch and does not call `model.forward` a second time.
Normalized internal token lists are borrowed across model, stepper, token-score,
and engine layers; only an external non-list sequence is normalized once.
Direct worker scoring and compact logits are backend capabilities, not
replacement estimator logic.
Real request-level token-score mechanics live in `zo_vllm.core.token_scores`.
Trainer-computed scalar objectives use `ProbeLossResult`, which stores one loss
per complete copy of the original batch. It never fabricates request NLL or
zero-token weights to reuse the token-score representation.

## Optimizer and Scheduler Boundary

The HF-native path uses the scheduler created by `Trainer.create_scheduler()`.
`ZOTrainer.create_optimizer()` creates a real `ZOSGDOptimizer`. The training
step estimates a `ZOGradientEstimate` and stages one `ZOPendingStep`; it does not
mutate model weights. Transformers invokes `ZOSGDOptimizer.step()` between its
native `on_pre_optimizer_step` and `on_optimizer_step` callbacks. The optimizer
reads LR and weight decay from its parameter group and consumes the pending step
exactly once. `ZOUpdateState` executes that update inside vLLM.

HF therefore owns scheduler type, warmup, stepping, checkpoint save, and
checkpoint restore. `zo_applied_learning_rate` reports the value used for the
current ZO update. Transformers records its standard `learning_rate` before
advancing the scheduler, so it has the same current-step value. The lower-level
runtime objects expose estimation only; callers must stage the returned
`ZOPendingStep` in an optimizer rather than applying an update through the
model or stepper.

Every loadable native or LoRA-bank checkpoint records the complete
`hf_to_vllm_mapping`, packed `hf_to_slice` mapping, and a deterministic
fingerprint. Resume compares that manifest with the current `WeightSync` before
loading tensors. Model names and target-module lists are provenance, not a
substitute for the exact mapping.

Native checkpoints use bounded per-rank safetensor parts. Live reload validates
the complete shard set, tensor keys, and tensor shapes before copying the first
model tensor. Checkpoint scope fields remain descriptive provenance; model
loading never turns them into implicit runtime configuration. Generic HF
callbacks depend only on injected observation protocols, not experiment runner
implementations.

Only exact ZO-SGD is currently supported. `optim` must be `sgd`, autograd
gradient clipping is disabled, and HF gradient accumulation remains restricted
to one microbatch per optimizer step. Accumulated low-rank backends reject
nonzero weight decay until an exact lazy-decay implementation exists.

## Training Step

`ZOTrainer.training_step` receives a ragged CPU list batch from the HF
dataloader. It deliberately does not call HF `_prepare_inputs`: vLLM owns device
placement. The model borrows normalized token rows, then exposes a post-forward
loss callback:

```text
probe_outputs = ZOTrainerModel.score_token_groups(runtime_probe)
probe_loss = ZOTrainer.compute_loss_from_outputs(probe_outputs)
```

The scorer is passed to `zo_model.estimate_with_score_fn(...)`. The stepper gives the
configured estimator both the minimal scorer and the real engine; only the
engine expands opaque probe slots into physical runtime IDs. The lower runtime
then:

1. samples the directions required by the configured estimator;
2. writes plus/minus or one-sided LoRA slots internally;
3. scores runtime-selected probes into compact causal logits or option NLL;
4. splits compact outputs by perturbation group and asks Trainer to compute one
   loss for each group;
5. builds a `ZOGradientEstimate` with explicit directions and coefficient;
6. returns a typed `ZOPendingStep` to `ZOTrainer`;
7. stages that estimate on `ZOSGDOptimizer`;
8. applies it through `UpdateState` when HF calls `optimizer.step()`.

Every estimator also sets `ZOEstimate.reported_loss` explicitly. Antithetic and
two-sided estimators report the mean probe objective, one-sided estimators
report the clean objective, and evolution strategies report negative mean
reward. `ZOTrainer` accepts only a typed `ZOPendingStep`; `ZOSGDOptimizer`
produces the typed `ZOStepResult` after the real update. Neither layer guesses a
loss from optional metric names or substitutes zero.

This means HF owns batch-to-loss semantics and ZO owns only the perturb/update
semantics. A combined plus/minus or multi-query vLLM forward must not compute one
loss over all probes. The adapter carries every Trainer-computed scalar in
`ProbeLossResult`, so estimators consume HF loss while vLLM still executes one
batched forward.

## Validation Surface

The ragged, device-resident path is covered by CPU contract tests and GPU
runtime-vs-HF probes. The GPU suite covers SST-2, BoolQ, SQuAD, and every supported
SuperGLUE objective. The current 11-task run passed all clean-loss and first-probe
checks; maximum clean-loss difference was `2.38e-7` and maximum probe difference
was `0.002558` with a fixed `0.005` probe tolerance. Artifacts live under
`phase4/results/hf_ragged_alignment_20260710_080625/`.

## Metrics and Callbacks

`ZOTrainer` uses Hugging Face's native training loop for logging, evaluation,
saving, and callback dispatch. Generic timing and quality metrics should stay in
the same places as a normal `Trainer` run:

- `train_runtime`, `train_steps_per_second`, and `train_samples_per_second`
  come from Hugging Face `train()`;
- `eval_runtime`, `eval_steps_per_second`, `eval_samples_per_second`, and
  `eval_loss` come from Hugging Face `evaluate()`;
- task metrics such as `eval_accuracy` come from the user-provided
  `compute_metrics` callable.

ZO-specific train-step metrics are exposed only as generic runtime observations.
`ZOStepResult.metrics()` is the boundary for these values, and `ZOTrainer.log`
adds them to Hugging Face logs with a `zo_` prefix, for example:

- `zo_loss_plus`, `zo_loss_minus`, `zo_projected_grad` when defined, and
  `zo_update_scale`;
- `zo_profile_set_plus_minus_directions_s`;
- `zo_profile_score_token_groups_s`;
- `zo_profile_scorer_batch_unpack_s`;
- `zo_profile_scorer_loss_dispatch_s`;
- `zo_profile_scorer_loss_to_host_s`;
- `zo_profile_scorer_forward_unpack_s`;
- `zo_profile_scorer_engine_call_s`;
- `zo_profile_scorer_output_postprocess_s`;
- `zo_profile_scorer_total_s`;
- `zo_profile_score_plus_minus_total_s`;
- `zo_update_fold_s`, `zo_update_accumulate_s`, and related update metrics.

`zo_profile_score_token_groups_s` remains the stable aggregate interval around
the scorer callback. The `zo_profile_scorer_*` fields split that interval into
one-time HF batch unpacking, the host-side engine call, output postprocessing,
HF loss dispatch, and the required scalar synchronization. The host engine call
is not labeled as GPU time. Device-resident runs resolve worker CUDA events only
after the grouped loss scalar reaches CPU and expose those values as
`zo_profile_worker_cuda_model_forward_s` and `zo_profile_worker_cuda_loss_s`.

The fields are nested rather than additive across the whole list:

```text
scorer_total
  scorer_batch_unpack
  scorer_forward_total
    scorer_engine_call
    scorer_output_postprocess
  scorer_loss_dispatch
  scorer_loss_to_host
```

Small residuals at each level cover Python dispatch and timer boundaries.

These fields may be consumed by Hugging Face reporters, W&B, TensorBoard,
`trainer_state.json`, or external scripts through normal HF log history. The
generic trainer must not implement phase-specific summaries such as
`tail_100.step_s.mean`, speedup tables, convergence reports, or custom result
JSON formats. Those belong in the corresponding `phase3/` or `phase4/` scripts
and collectors.

Hugging Face `TrainerCallback` remains the extension point for experiment-local
behavior. A phase script may pass its own callback for per-step wall timing,
tail-window summaries, extra artifact writing, or reporting policy. Such
callbacks should live with the phase script unless they are broadly useful
outside a specific experiment protocol.

## Checkpointing

Checkpoint cadence remains Hugging Face `Trainer` behavior. Runtime payload
writing is delegated to `ZOVLLMCheckpointHandler`:

- `metadata`: trainer/runtime metadata only, not loadable;
- `native`: full effective vLLM model checkpoint, loadable;
- `lora`: resumable LoRA-bank training state.

`load_best_model_at_end` requires `zo_checkpoint_mode="native"` because metadata
and LoRA-bank payloads are not full effective model checkpoints.

`ZOTrainer` delegates to Hugging Face's native `_save_checkpoint` flow. HF owns
Trainer state, RNG state, stateful callbacks, checkpoint rotation, and Hub
integration. Its single `save_model(..., _internal_call=True)` call invokes the
runtime payload handler once with `TrainerState.global_step`, then writes strict
ZO metadata. Loadable checkpoints also persist direction-provider state
independently from model weights. After loading into rebuilt workers, Trainer
invalidates the device-slot state so the first resumed probe copies the restored
V basis into the new LoRA slots without resampling it. The handler is owned by
Trainer, and its mode must exactly match
`ZOTrainerArguments.zo_checkpoint_mode`. Missing metadata, missing handlers for
loadable modes, and non-loadable resume attempts are hard errors. Reusable
payload operations live under `zo_vllm.training`.

## Success Criteria

`zo_trainer` is successful when a user can take a normal Hugging Face tokenized
training script, replace the model/trainer with the ZO-vLLM facade, and keep the
dataset, tokenizer, collator, metrics, callback, logging, and checkpoint cadence
as native Hugging Face code.
