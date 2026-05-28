# LLM Zeroth-Order Fine-Tuning is an Inference Workload

This repository contains the code for:

**LLM Zeroth-Order Fine-Tuning is an Inference Workload**<br>
Zelin Li and Caiwen Ding<br>
arXiv: [2605.28760](https://arxiv.org/abs/2605.28760)<br>
DOI: [10.48550/arXiv.2605.28760](https://doi.org/10.48550/arXiv.2605.28760)

The project reorganizes LLM zeroth-order (ZO) fine-tuning as an
inference-style scoring workload. Instead of executing repeated plus/minus
objective evaluations inside a conventional training loop, ZO-vLLM routes the
dominant scoring phase through a vLLM-based runtime and represents nearby
parameter states as dynamic LoRA adapter states.

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
  experiment/            Shared experiment utilities
phase1/                  Perturbation/sign alignment checks
phase2/                  Baseline alignment and short convergence validation
phase3/                  Core-step speed and scaling experiments
phase4/                  Long-run OPT-13B convergence experiments
phase6/                  MeZO-style high-rank factorized ZO experiment notes
third_party/
  vllm/                  vLLM fork with direct ZO scoring support
  LOZO/                  Official LOZO/MeZO baseline fork
```

Important runtime components:

- `zo_vllm/core/direct_worker_scorer.py`: direct vLLM worker scoring,
  plus/minus loss splitting, and SST-2 option-loss scoring.
- `zo_vllm/core/temp_lora_runtime.py`: GPU-resident temporary plus/minus LoRA
  slots.
- `zo_vllm/core/lozo_controller.py`: direction sampling and factorized ZO
  state.
- `zo_vllm/core/weight_sync.py`: worker-local low-rank base-weight updates.

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

The main repository uses Python 3.12 through `uv`:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Build the vLLM fork:

```bash
MAX_JOBS=16 NVCC_THREADS=4 VLLM_TARGET_DEVICE=cuda \
  uv pip install -e third_party/vllm --no-build-isolation
```

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
export VLLM_BATCH_INVARIANT=0
```

## Quick Smoke Checks

These commands check that the public entrypoints start and parse arguments:

```bash
.venv/bin/python phase3/runners/train_vllm_perf.py --help
third_party/LOZO/large_models/.venv/bin/python \
  phase3/runners/train_lozo_baseline_perf.py --help
.venv/bin/python phase4/runners/run_phase4_job.py --help
```

Dry-run launchers without starting long experiments:

```bash
.venv/bin/python phase3/runners/launch_scaling_sweep.py \
  --lozo-gpu 4 --vllm-gpu 5 --models facebook/opt-1.3b \
  --batch-sizes 16 --steps 10 --num-samples 16 --dry-run

.venv/bin/python phase4/runners/launch_phase4.py \
  --config phase4/configs/phase4_sst2_opt13b_paired_official_vllm_grid.json \
  --gpus 6,7 --dry-run
```

## Example vLLM ZO Run

Use a small run first to verify the local build and GPU setup:

```bash
CUDA_VISIBLE_DEVICES=4 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
VLLM_BATCH_INVARIANT=0 \
.venv/bin/python -u phase3/runners/train_vllm_perf.py \
  --model-name facebook/opt-125m \
  --steps 5 \
  --warmup-steps 0 \
  --batch-size 1 \
  --num-samples 8 \
  --num-dev 4 \
  --rank 2 \
  --lr 1e-7 \
  --eps 1e-3 \
  --step-interval 50 \
  --eval-interval 0 \
  --train-objective sst2_classification \
  --zo-random-device cuda \
  --direction-sampling flat \
  --train-scope lora_only \
  --batch-invariant 0 \
  --enforce-eager 0 \
  --lora-residency gpu \
  --lora-injection direct \
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

## Reproducing Main Experiments

Phase 3 speed/scaling:

```bash
.venv/bin/python phase3/runners/launch_scaling_sweep.py \
  --lozo-gpu 4 \
  --vllm-gpu 5 \
  --models facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,facebook/opt-13b \
  --batch-sizes 16,32,64,128 \
  --steps 300 \
  --warmup-steps 5 \
  --num-samples 1000
```

Phase 4 long-run convergence uses JSON configs under `phase4/configs/`:

```bash
.venv/bin/python phase4/runners/launch_phase4.py \
  --config phase4/configs/phase4_sst2_opt13b_paired_official_vllm_grid.json \
  --gpus 4,5,6,7
```

Collectors:

```bash
.venv/bin/python phase3/runners/collect_scaling_sweep.py <phase3-run-dir>
.venv/bin/python phase4/runners/collect_phase4_summary.py <phase4-run-dir>
```

## Results in This Repository

- `phase1/phase1_results.md`: vLLM LoRA perturbation sign alignment.
- `phase2/README.md`: short-horizon LOZO/vLLM alignment checks.
- `phase3/README.md`: core-step performance experiments.
- `phase4/README.md`: long-run convergence summaries.
- `phase6/README.md`: MeZO-style high-rank factorized ZO comparison.

Generated results and presentation packages are intentionally ignored by git:

- `phase*/results/`
- `preprint_package/`
- `preprint_package.zip`

## Citation

```bibtex
@misc{li2026llmzerothorderfinetuninginference,
  title        = {LLM Zeroth-Order Fine-Tuning is an Inference Workload},
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
