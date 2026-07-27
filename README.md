# LLM Zeroth-Order Optimization is an Inference Workload

This repository contains the code for:

**LLM Zeroth-Order Optimization is an Inference Workload**<br>
Zelin Li and Caiwen Ding<br>
arXiv: [2605.28760](https://arxiv.org/abs/2605.28760)<br>
DOI: [10.48550/arXiv.2605.28760](https://doi.org/10.48550/arXiv.2605.28760)

The project reorganizes LLM zeroth-order (ZO) fine-tuning as an
inference-style scoring workload. Instead of executing repeated plus/minus
objective evaluations inside a conventional training loop, ZO-vLLM routes the
dominant scoring phase through a vLLM-based runtime and represents nearby
parameter states as dynamic LoRA adapter states.

## Installation

The supported environment is Python 3.12.13 with uv 0.11.15. Install exactly
the checked-in resolution:

```bash
git submodule update --init --recursive
uv sync --locked
uv lock --check
```

Normal installation keeps `third_party/vllm` Python sources editable and uses
the pinned upstream precompiled native wheel. Do not compile vLLM unless the
fork changes C++, CUDA, CMake, or generated native interfaces. FastAPI and
Starlette are transitive vLLM dependencies; the ZO serving router composes the
vLLM FastAPI lifespan and does not require `starlette<1`.

## Highlights

- **OPT-13B SST-2 long run**: vLLM completes the matched LoRA-only 20k-step
  LoZO run in an estimated 0.51 training hours versus 4.15 hours for the
  official LoZO baseline, a **8.13x speedup**.
- **Convergence**: the vLLM path reaches `0.922` final evaluation accuracy and
  `0.931` final full-validation accuracy in the OPT-13B run.
- **Scaling**: core-step experiments across OPT-1.3B to OPT-13B show
  `2.34x` to `7.72x` speedups.
- **MeZO-style experiment**: high-rank factorized ZO tracks a MeZO-like loss
  trajectory while running up to `2.55x` faster.

## Repository Layout

```text
zo_vllm/
  core/                  Shared ZO-vLLM runtime components
  analysis/              Pre-training AGZO subspace exploration utilities
  training/              Reusable ZO training state utilities
  experiment/            Shared experiment utilities
phase1/                  Perturbation/sign alignment checks
phase2/                  Baseline alignment and short convergence validation
phase3/                  Core-step speed and scaling experiments
phase4/                  Long-run OPT-13B convergence experiments
zo_post/                 Post-phase AGZO experiments and worker AGZO probes
phase6/                  MeZO-style high-rank factorized ZO experiment notes
phase7/                  Quantized base + LoRA update bank experiment scripts
third_party/
  vllm/                  vLLM fork with direct ZO scoring support
  LOZO/                  Official LOZO/MeZO baseline fork
```

Important runtime components:

- `zo_vllm/core/direct_worker_scorer.py`: direct vLLM worker scoring,
  plus/minus loss splitting, and task-agnostic option-loss scoring.
- `zo_vllm/core/token_scores.py`: request-level token-score chunking,
  merging, and slicing for real token-NLL results.
- `zo_vllm/core/probe_results.py`: objective-level scalar losses returned by
  Hugging Face loss-backed probes without emulating token NLL.
- `zo_vllm/core/lora_runtime/`: GPU-resident plus/minus LoRA update
  slots.
- `zo_vllm/training/direction/`: direction sampling and factorized ZO
  state.
- `zo_vllm/core/weight_sync.py`: worker-local low-rank base-weight updates.
- `zo_vllm/analysis`: AGZO V-subspace collection and rank-aware similarity
  summaries for pre-training exploration.
- `zo_vllm/training/direction/subspace_queue.py`: fixed-shape AGZO subspace queue for
  reusing recent activation subspaces without task-specific code.
- `zo_vllm/core/perturbation_normalization.py`: analytic perturbation-energy
  normalization shared by LOZO, AGZO, UAGZO, and serving-time worker updates.
- `zo_vllm/training/update_state.py`: accumulated low-rank update state. In
  accumulate mode, clean train/eval/final scoring sees `W + U_accum V^T`
  through a temporary LoRA slot; base weights fold only when `nu` refreshes the
  direction basis. Phase 7 bank mode keeps quantized/prequantized base weights
  read-only and routes all high-precision updates through LoRA bank slots.
- `zo_vllm/training/worker_update_bank.py`: worker-resident update backend for
  serving-time ZO. It owns direction sampling, LoRA-bank update state, slot
  preparation, and scalar update apply inside the vLLM worker.
- `zo_vllm/serving/`: serving lifecycle and scheduled runtime backend. It submits
  background compact-NLL scoring requests through the vLLM scheduler, delegates
  update state to the worker update bank, and exposes `/zo_vllm/serving_zo/*` only
  when `VLLM_ZO_SERVING_TRAINING=1`. Slot writes default to a conservative
  per-pair barrier: write plus/minus slots on a background CUDA stream, wait for
  the copy event before submitting the matching low-priority scoring pair, and
  do not overwrite plus/minus slots until that pair returns. The HF trainer thread
  supports conservative `idle_gap` score admission and a more aggressive
  `scheduler_only` mode. `idle_gap` waits only before score submission until the
  foreground load tracker reports an empty gap; `scheduler_only` submits the
  low-priority score pair immediately after slot writes complete and lets the
  vLLM scheduler order the work. In compile mode, batches that contain serving
  ZO LoRA compact-NLL scoring skip the compiled wrapper and CUDA graph locally;
  foreground-only serving batches still use the server's normal compile path.
  Direct LoRA scope follows vLLM `target_modules`; optional `lm_head` and
  `embed_tokens` support is resolved into that same list before metadata,
  slot registration, and worker-bank validation. For tied input/output
  embeddings, vLLM LoRA still needs both paths registered: `embed_tokens` covers
  token lookup, while `lm_head` routes logits through `LogitsProcessorWithLoRA`.
  Base-weight updates remain tied and should update the shared embedding matrix
  only once.
- `zo_vllm/training/optimizer.py`: shared staged ZO-SGD state machine used by
  offline direct scoring and serving scheduled scoring backends.
- `zo_vllm/experiment/scoring/sst2.py`: SST-2-specific wrappers built
  on the shared direct-worker API.
- `zo_vllm/tasks/superglue/`: LOZO-style SuperGLUE prompt adapters split by
  independently trainable task: BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, and
  WSC.fixed. The corresponding vLLM objectives are named
  `superglue_<task>_classification`.
- `zo_post/`: experiment-only AGZO follow-up area. Runners and notes live here;
  large run outputs stay under ignored `zo_post/results/`, centralized stdout
  logs stay under ignored `zo_post/logs/`, and worker-probe artifacts stay
  under ignored `zo_post/artifacts/`.

## Environment

Clone with submodules:

```bash
git clone --recursive https://github.com/playeriv65/zo-vllm.git
cd zo-vllm
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the Python 3.12 environment from the checked-in lock:

```bash
uv sync --locked
uv lock --check
```

This keeps the vLLM Python fork editable and obtains its native extensions from
the pinned precompiled wheel. Do not run a source build for normal deployment.

The official LOZO baseline uses its own environment:

```bash
cd third_party/LOZO/large_models
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

The local experiments assume:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

## OPT Tokenizer Policy

OPT's HuggingFace/vLLM-native tokenizer uses `</s>` (token id `2`) as the
beginning-of-sequence token. The official MeZO/LOZO code path changes OPT to
use `<s>` (token id `0`) for its baseline experiments. ZO-vLLM therefore keeps
the native OPT tokenizer by default and exposes an explicit reproduction switch:

```bash
--opt-bos-mode native  # default, HF/vLLM OPT behavior
--opt-bos-mode lozo    # strict MeZO/LOZO OPT reproduction behavior
```

Use `lozo` only when comparing loss or convergence directly against MeZO/LOZO
OPT baselines. System-level vLLM comparisons that are internally self-consistent
can use the native tokenizer. Phase 1 alignment checks compare HF/vLLM behavior
under the same token IDs, so they do not by themselves validate the MeZO/LOZO
tokenizer policy.

## Checkpoint Policy

Training runners expose Hugging Face-like checkpoint controls:

```bash
--save-strategy no|steps|best
--save-steps 1000
--save-total-limit 3
--metric-for-best-model eval_loss
--greater-is-better auto
--load-best-model-at-end 0|1
--save-checkpoint-mode auto|metadata|native|lora
```

`steps` saves every `--save-steps` measured training steps. `best` saves only
when the selected eval metric improves. `native` writes vLLM sharded full-model
checkpoints and is required for `--load-best-model-at-end 1`. `lora` writes a
lightweight LoRA-bank state for training resume with `--resume-lora-checkpoint`;
it does not replace full-model checkpoints.

## Training Layering

ZO-vLLM separates native Hugging Face training control from ZO/vLLM runtime
semantics:

```text
Hugging Face Dataset / DataLoader / TrainerCallback / compute_metrics
  -> ZOTrainer / ZOTrainerArguments
     -> ZOTrainerModel
        -> VLLMZOModel.estimate_with_score_fn()
           -> ZOStepper
              -> DirectionProvider
              -> ZOEstimator
              -> ZOGradientEstimate / ZOPendingStep
     -> ZOSGDOptimizer.step()
        -> UpdateState
           -> ZOVLLMEngine / vLLM worker / LoRA runtime
```

ES generation-reward tasks use `ZORolloutTrainerModel` at the same model
boundary. The task passes a reward function and HF `compute_metrics`; the
standard `ZOTrainer.train()` and `ZOTrainer.evaluate()` loops remain unchanged.
Population size, reward shaping, direction mode, and query microbatching remain
owned by the configured estimator below the Trainer.

Hugging Face owns datasets, preprocessing, sampling, batching, loss, metrics,
callbacks, logging, evaluation, and checkpoint cadence. `ZOTrainerModel`
converts one ragged HF batch into compact causal logits or option logits without
choosing perturbation slots. `ZOEstimator` owns antithetic, one-sided,
multi-query, and evolution-strategy probe semantics. `ZOVLLMEngine` owns
physical LoRA IDs and backend execution. New tasks must enter through HF fields
such as `input_ids`, `attention_mask`, and `labels`; they must not add task
branches to Trainer.

### Callback Surface

Use the native Hugging Face `TrainerCallback` surface for outer-loop behavior:

```text
on_init_end
on_train_begin / on_train_end
on_epoch_begin / on_epoch_end
on_step_begin / on_step_end / on_substep_end
on_pre_optimizer_step / on_optimizer_step
on_prediction_step / on_predict
on_evaluate / on_save / on_log
```

Callbacks receive Hugging Face `TrainingArguments`, `TrainerState`, and
`TrainerControl`. More specific post-score/pre-update logic belongs to
`ZOStepper` callbacks:
`on_direction_sampled`, `on_score_end`, `on_update_begin`, and `on_update_end`.
Use those hooks for algorithm-level behavior such as inspecting the estimated
direction or intervening before `UpdateState.apply()`. Do not add such behavior
as task-runner branches or trainer flags.

## Using ZO-vLLM from Another Repository

Install the vLLM fork first, then install this repository as an editable
dependency:

```bash
uv pip install -e /path/to/zo-vllm/third_party/vllm --no-build-isolation
uv pip install -e /path/to/zo-vllm
```

The public engine API is `zo_vllm.ZOVLLMEngine`. External experiments should
pass their model, token IDs, objective aggregation, and output paths from their
own config or CLI. The engine owns vLLM startup, persistent plus/minus LoRA
slots, direct worker scoring, and direction updates. Training-state utilities
such as `SubspaceQueue` live under `zo_vllm.training`; optimizer and scheduler
policy belongs to Hugging Face `TrainingArguments`:

```python
from transformers import AutoConfig
from zo_vllm import SubspaceQueue, ZOVLLMEngine

model_name = cfg.model_name
model_config = AutoConfig.from_pretrained(model_name)

def objective(score):
    # Convert per-request mean NLLs into the task loss used by this experiment.
    return my_task_loss(score.request_mean_nll, labels)

with ZOVLLMEngine(
    model=model_name,
    rank=cfg.rank,
    model_config=model_config,
    max_model_len=cfg.max_length,
    gpu_memory_utilization=cfg.gpu_memory_utilization,
) as engine:
    result = engine.score_plus_minus_directions(
        directions,
        token_id_groups,
        eps=cfg.eps,
        labels=token_labels,
        objective=objective,
        max_logits_tokens=cfg.max_logits_tokens,
    )

print(result.loss_plus, result.loss_minus, result.projected_grad)
```

For training, use the `zo_trainer` package. It builds directly from
`VLLMZOModel`, the HF model facade, `ZOStepper`, and runtime checkpoint helpers;
it does not wrap another trainer.

The `zo_trainer` data path is ordinary Hugging Face code:

```python
raw = load_dataset(...)
tokenized = raw.map(preprocess_function, batched=True, remove_columns=raw.column_names)
collator = VLLMDataCollator()
trainer = ZOTrainer(
    model=zo_vllm_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
)
trainer.train()
```

The preprocess function should emit standard HF fields such as `input_ids`,
`attention_mask`, and `labels`; ignored loss positions use `labels=-100`.
`zo_trainer.preprocessing` includes generic `Dataset.map` helpers for causal-LM
and prompt-target LM examples. Task-specific prompt and verbalizer helpers live
under `zo_vllm.tasks.hf_preprocessing`; neither layer creates trainer branches.
`VLLMDataCollator` preserves ragged CPU token lists because vLLM does not need
sequence padding.

During training, `ZOTrainer.training_step` exposes Hugging Face loss computation
to the runtime without choosing LoRA slots itself. The configured ZO estimator
still drives `score_plus_minus_directions`, `score_token_groups`, and
opaque `set_probe_directions` / `score_probe_slots` operations through a
minimal objective scorer beside the real engine, so two-sided, one-sided, and multi-query probing
stay runtime-owned without exposing LoRA IDs. The vLLM backend returns
compact logits for the active causal-LM label positions, and `ZOTrainer`
computes loss through the Transformers causal-LM loss helper. The backend intentionally
does not materialize full `[batch, seq, vocab]` logits for masked and padded
positions.

The `zo_trainer` ownership split is:

| Layer | Responsibility |
| --- | --- |
| Hugging Face script | Dataset loading, `Dataset.map`, tokenizer/processor, collator, `compute_metrics`, callbacks |
| `ZOTrainer` | Hugging Face lifecycle, native LR scheduler, estimate staging, and exposing `Trainer.compute_loss` to the runtime |
| `ZOTrainerModel` | HF batch facade and compact-logits/request-NLL forward pass |
| `ZOStepper` | Direction sampling, callback ordering, estimate construction, and runtime apply orchestration |
| `ZOEstimator` | Probe algorithm: antithetic, one-sided, multi-query, ES, and aggregation |
| `ZOSGDOptimizer` | Native HF optimizer step, LR/weight decay, and staged-estimate consumption |
| `ZOUpdateState` | vLLM weight mutation, low-rank accumulation, and fold execution |
| `zo_vllm.core.probe_results` | Complete HF objective-group scalar losses |
| `zo_vllm.core.token_scores` | Real request-level token-score chunking, merging, and slicing |
| `ZOVLLMEngine` | LoRA slot writes, direct scoring/logits, vLLM runtime checkpoint payloads |

Do not bypass the estimator from `zo_trainer`. Direct scoring/logits are engine
backend operations; estimator code owns when and how often those backend
operations are invoked.
Per-request token-score mechanics live in `zo_vllm.core.token_scores`.
Trainer-computed scalar losses use `ProbeLossResult` and remain distinct from
request NLL throughout estimator slicing.

The HF-native path uses `ZOSGDOptimizer` and the scheduler created by
Transformers. `training_step` stages one estimated gradient; the real update
runs inside HF's `optimizer.step()` callback boundary. The current learning
rate is read from the optimizer parameter group. `zo_applied_learning_rate` and
the standard HF `learning_rate` both describe that current optimizer step.
Only `optim=sgd` is supported; unsupported AdamW, autograd clipping, and
accumulated-backend weight decay fail explicitly.

Serving-time training runs the same synchronous HF `ZOTrainer`, stages the same
`ZOPendingStep`, and applies it through the same `ZOSGDOptimizer`. Only the
scheduled runtime backend bridges scoring and worker-bank mutation onto the
API-server event loop. Serving QoS admission remains outside the optimizer and
estimator layers.

Loadable HF checkpoints include the direction-provider cache in addition to the
runtime model, optimizer, scheduler, RNG, and Trainer state. Rebuilt vLLM
workers force one device-slot synchronization from that cache before the first
resumed probe; this preserves the sampled direction while avoiding empty LoRA
slots after resume.

`zo_trainer` keeps metric ownership aligned with a normal Hugging Face
`Trainer`. HF reports `train_runtime`, `train_steps_per_second`,
`eval_runtime`, `eval_steps_per_second`, `eval_loss`, and any user-supplied
`compute_metrics` values such as `eval_accuracy`. ZO runtime observations are
added to normal HF logs with a `zo_` prefix, including probe losses, projected
gradient, profile timings, and update timings. Phase-specific summaries such as
`tail_100.step_s.mean`, speedup tables, convergence reports, or custom result
JSON files belong in `phase3/` and `phase4/` scripts and collectors, not in the
generic `zo_trainer` package.

### Perturbation Normalization

ZO-vLLM samples factorized LoRA perturbations as `Delta W = scale * U @ V.T`.
Different direction providers use different internal parameterizations: plain
LOZO samples Gaussian `U` and Gaussian `V`, AGZO reuses activation-derived
unit-basis `V`, UAGZO/SUAGZO change how `U` is sampled, and queued AGZO
concatenates several recent `V` subspaces. Without a global convention, changing
`rank`, queue size, or provider can silently change the perturbation energy even
when `eps` and `direction_scale` stay fixed.

The default `perturbation_normalization="rms"` keeps the expected global RMS of
`U @ V.T` at `1` per affected weight:

```text
target_energy = sum_modules(out_features * in_features)
raw_energy    = sum_modules(out_features * effective_rank * E[||v_j||_2^2])
norm_scale    = sqrt(target_energy / raw_energy)
final_scale   = direction_scale * norm_scale
```

This is an analytic normalization. It does not compute `||U @ V.T||` or any
runtime matrix norms in the hot path. Providers attach only the metadata needed
for the formula:

- Gaussian `V`: `E[||v_j||_2^2] = in_features` unless `v_normalization="unit"`.
- AGZO `V`: `E[||v_j||_2^2] = 1` because the worker returns a unit subspace
  basis.
- Queued AGZO `V`: `effective_rank` counts only active queued subspaces; padded
  zero slots do not increase the expected energy.
- Pool/subspace `U`: columns are sampled on the same column-energy convention as
  default Gaussian `U`; rank and queue effects are handled by the global
  normalization, not by local `1/sqrt(rank)` factors.

So the default behavior is normalized to the energy scale of Gaussian LOZO, not
to the same distribution. AGZO and UAGZO still use their own structured
subspaces; they only share the perturbation RMS convention. Use
`perturbation_normalization="none"` to recover the raw legacy scale, and use
`direction_scale` as the explicit user amplitude multiplier after normalization.

### Pre-Training Subspace Exploration

Before launching a full AGZO run, external repositories can collect activation
subspaces with `zo_vllm.analysis` while keeping dataset loading and prompt
encoding local to the task repo:

```python
from zo_vllm.analysis import (
    SubspaceCollectionConfig,
    collect_v_subspaces,
    select_subspace_pairs,
    summarize_subspace_pairs,
)

records = collect_v_subspaces(
    engine,
    token_id_group_batches_by_subspace,
    config=SubspaceCollectionConfig(
        rank=cfg.rank,
        power_iter_steps=cfg.power_iter_steps,
        basis_method=cfg.basis_method,  # "power_iter" or "svd"
        summary_device="cuda",
    ),
)
v_maps = [record.v for record in records]
summaries = [
    summarize_subspace_pairs(v_maps, pairs, tag=tag, device="cuda").to_dict()
    for tag, pairs in select_subspace_pairs(
        len(records),
        mode="random_nonoverlap",
        random_pairs=cfg.random_pairs,
        seed=cfg.seed,
    )
]
```

`basis_method="power_iter"` matches the AGZO approximation used in training.
`basis_method="svd"` computes the exact right-singular subspace for one
activation batch, and for kappa-style chunked collection uses the exact
eigenspace of the equal-weighted per-batch activation covariance. Rank-1
similarity uses the exact principal cosine directly; forcing the full SVD
path gives the same value up to floating-point noise.

## Quick Smoke Checks

These commands check that the public entrypoints start and parse arguments:

```bash
.venv/bin/python -m zo_vllm.experiment.runners.vllm_zo_task --help
.venv/bin/python -m zo_vllm.experiment.runners.backend_job --help
.venv/bin/python -m zo_vllm.experiment.runners.job_queue_worker --help
.venv/bin/python -m zo_vllm.experiment.runners.collect_superglue_scaling --help
```

Dry-run launchers without starting long experiments:

```bash
.venv/bin/python phase3/runners/launch_superglue_scaling_sweep.py \
  --gpus 4,5 --tasks sst2 --models facebook/opt-1.3b \
  --batch-sizes 16 --steps 10 --num-samples 16 --num-dev 8 \
  --eval-interval 10 --dry-run

.venv/bin/python phase4/runners/launch_phase4.py \
  --config phase4/configs/phase4_sst2_opt13b_paired_official_vllm_grid.json \
  --gpus 6,7 --dry-run
```

## Test Layers

Unit and lightweight integration tests stay under `tests/test_*.py` and should
not launch real vLLM model runs:

```bash
uv run pytest tests
```

GPU end-to-end smoke tests live under `e2e_tests/`. They launch real scripts
with `facebook/opt-125m`, initialize vLLM, and run one training step, so they
are opt-in:

```bash
ZO_VLLM_RUN_E2E=1 CUDA_VISIBLE_DEVICES=0,1 \
uv run pytest e2e_tests -m e2e
```

## Example vLLM ZO Run

Use a small run first to verify the local build and GPU setup:

```bash
CUDA_VISIBLE_DEVICES=4 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
.venv/bin/python -u -m zo_vllm.experiment.runners.vllm_zo_task \
  --model-name facebook/opt-125m \
  --steps 5 \
  --warmup-steps 0 \
  --batch-size 1 \
  --num-samples 8 \
  --num-dev 4 \
  --rank 2 \
  --lr 1e-7 \
  --eps 1e-3 \
  --nu 50 \
  --eval-interval 0 \
  --train-objective sst2_classification \
  --zo-random-device cuda \
  --direction-sampling flat \
  --train-scope lora_normal \
  --enforce-eager 0 \
  --weight-update direct \
  --weight-update-precision param \
  --direct-update-mode accumulate \
  --qkv-weight-update batched \
  --sync-weight-update 0 \
  --scoring-backend direct_worker \
  --direct-worker-max-logits-tokens 8192 \
  --direct-worker-loss-impl logprobs \
  --direct-lora-from-directions 1 \
  --base-eval-mode skip \
  --accuracy-eval-mode skip \
  --progress-interval 1 \
  --train-loss-interval 0 \
  --gpu-memory-utilization 0.3 \
  --output-dir /tmp/zo_vllm_smoke
```

`--train-scope lora_normal` is the canonical normal LoRA perturbation scope.
Reports and experiment names should use these labels:

| Report label | Meaning |
|---|---|
| `lora_normal` | vLLM/LOZO normal LoRA scope for attention/MLP modules, without token embedding or tied LM head perturbations. |
| `lora_embed_input_only` | Diagnostic scope that also perturbs input token embeddings but does not register the tied LM head path. |
| `lora_embed_tied_head` | Legacy diagnostic label for `lora_normal --perturb-embeddings 1` on tied-embedding models. |
| `lora_full` | All vLLM LoRA-compatible targets: normal linear LoRA targets plus token embeddings and the tied LM head logits path when the model ties input/output embeddings. |
| `mezo_full` | Third-party MeZO full-parameter perturbation baseline. |

## Reproducing Main Experiments

Phase 3 speed/scaling:

```bash
.venv/bin/python phase3/runners/launch_superglue_scaling_sweep.py \
  --gpus 4,5 \
  --tasks sst2,superglue_boolq,superglue_cb,superglue_copa,superglue_multirc,superglue_record,superglue_rte,superglue_wic,superglue_wsc \
  --models facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,facebook/opt-13b \
  --batch-sizes 16,32,64,128 \
  --steps 300 \
  --warmup-steps 20 \
  --num-samples 1000 \
  --num-dev 500 \
  --eval-interval 300
```

Phase 4 long-run convergence uses JSON configs under `phase4/configs/`:

```bash
.venv/bin/python phase4/runners/launch_phase4.py \
  --config phase4/configs/phase4_sst2_opt13b_paired_official_vllm_grid.json \
  --gpus 4,5,6,7
```

Collectors:

```bash
.venv/bin/python -m zo_vllm.experiment.runners.collect_superglue_scaling <phase3-run-dir>
.venv/bin/python phase4/runners/collect_phase4_summary.py <phase4-run-dir>
```

## Results in This Repository

- `phase1/phase1_results.md`: vLLM LoRA perturbation sign alignment.
- `phase2/README.md`: short-horizon LOZO/vLLM alignment checks.
- `phase3/README.md`: core-step performance experiments.
- `phase4/README.md`: long-run convergence summaries.
- `phase6/README.md`: MeZO-style high-rank factorized ZO comparison.
- `docs/HISTORY_REWORK.md`: archive tags and semantic branch reconstruction.

Generated results and presentation packages are intentionally ignored by git:

- `phase*/results/`
- `preprint_package/`
- `preprint_package.zip`

## Citation

```bibtex
@misc{li2026llmzerothorderfinetuninginference,
  title        = {LLM Zeroth-Order Optimization is an Inference Workload},
  author       = {Zelin Li and Caiwen Ding},
  year         = {2026},
  eprint       = {2605.28760},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  doi          = {10.48550/arXiv.2605.28760}
}
```

## License

This repository contains project code plus third-party submodules. See the
license files in each third-party component for their respective terms.
