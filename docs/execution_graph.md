# ZO-vLLM Execution Graph

This document records the current Hugging Face-native training and runtime
boundaries.

## Training Data Path

```text
Hugging Face Dataset
  -> Dataset.map(tokenizer / processor / chat template)
     -> input_ids / attention_mask / labels
  -> Hugging Face sampler and DataLoader
  -> VLLMDataCollator
     -> ragged CPU lists, no padding and no tokenization
  -> ZOTrainer.training_step()
```

`ZOTrainerArguments.use_cpu=True` keeps Hugging Face orchestration on CPU. vLLM
owns GPU placement and execution.

## ZO Train Step

```text
ZOTrainer.training_step()
  -> ZOTrainerModel.zo_estimate(HF batch)
     -> one conversion to TokenProbeBatch
     -> VLLMZOModel.estimate_with_score_fn()
        -> ZOStepper
           -> DirectionProvider
           -> ZOEstimator
              -> opaque RuntimeProbeSlots
              -> ZOVLLMEngine.score_*_with_scorer()
                 -> vLLM direct worker
                    -> compact active-token logits, or
                    -> device-resident request NLL
              -> ZOTrainer.compute_loss_from_outputs()
                 -> Transformers causal-LM loss, or
                 -> Transformers fixed cross entropy
              -> ProbeLossResult
              -> ZOGradientEstimate
        -> typed ZOPendingStep with explicit reported_loss
  -> ZOSGDOptimizer.stage()
Hugging Face on_pre_optimizer_step
  -> ZOSGDOptimizer.step()
     -> read current LR and weight decay from the optimizer parameter group
     -> ZOPendingStep.apply()
        -> UpdateState.apply()
        -> typed ZOStepResult
Hugging Face on_optimizer_step
  -> scheduler.step()
```

The estimator owns antithetic, one-sided, multi-query, and evolution-strategy
probe plans. `ZOTrainer` never selects perturbation sides or physical LoRA IDs.
The engine owns slot allocation and worker execution. Hugging Face owns the
objective loss.

Transformers owns optimizer/scheduler ordering, warmup, stepping, callback
dispatch, and checkpoint restore. `ZOSGDOptimizer` performs the real update and
reports its LR as `zo_applied_learning_rate`. The estimator never receives LR or
weight decay.

## Evaluation Path

```text
Hugging Face evaluation loop
  -> ZOTrainer.prediction_step()
  -> ZOTrainerModel.forward()
  -> ZOVLLMEngine clean forward
  -> compact logits and aligned labels
  -> ZOTrainer.compute_loss_from_outputs()
  -> eval_loss / compute_metrics / TrainerCallback
```

No training task uses a separate raw-row evaluator. Task-specific preprocessing
must finish before the batch reaches Trainer.

## Checkpoint Path

```text
Hugging Face Trainer._save_checkpoint()
  -> ZOTrainer.save_model(..., _internal_call=True)
     -> ZOVLLMCheckpointHandler.save_checkpoint()
     -> zo_direction_state.pt
     -> strict zo_checkpoint_metadata.json
  -> optimizer and scheduler state
  -> RNG state
  -> stateful callback state
  -> trainer_state.json
  -> checkpoint rotation / Hub integration
```

`ZOTrainer` owns the runtime handler. Handler mode must match
`ZOTrainerArguments.zo_checkpoint_mode`. Metadata-only checkpoints cannot be
resumed. Missing metadata and non-loadable payloads are hard errors.
On load, the restored mathematical direction cache remains unchanged, while the
next probe forces one V-basis copy into newly created runtime LoRA slots.

## Serving Path

```text
ServingZOTrainer
  -> AsyncWorkerUpdateBankClient
  -> vLLM worker RPC
     -> worker-resident DirectionProvider and UpdateState
     -> ZOVLLMEngine-compatible scoring/runtime operations
```

Serving admission and worker residency are runtime concerns. They do not add
dataset, loss, or Trainer policy to the core engine.

## Ownership Rules

- Hugging Face owns datasets, tokenization, sampling, DataLoader orchestration,
  loss, metrics, callbacks, logging, evaluation, and checkpoint cadence.
- `zo_trainer` owns only the native Trainer/model/runtime adaptation boundary.
- `ZOTrainerModel` owns HF batch conversion and compact model outputs.
- `ZOStepper` owns ZO callback order and estimate/runtime orchestration.
- `ZOEstimator` owns probe planning and aggregation.
- `ZOSGDOptimizer` owns the HF optimizer protocol and ZO-SGD policy.
- `DirectionProvider` owns direction generation.
- `UpdateState` owns runtime weight mutation and accumulation/fold behavior.
- `ZOVLLMEngine` owns LoRA slots, direct worker calls, and runtime capabilities.
- Phase directories own benchmark parameters, summaries, and result artifacts;
  they must not add phase-specific behavior to `zo_trainer`.
