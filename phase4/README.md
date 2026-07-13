# Phase 4: Long-Run Convergence on Real Training Tasks

Phase 4 evaluates long-run convergence and end effectiveness of LoZO-vLLM on
real training workloads, starting with OPT small model + SST-2.

## Scope

- Baseline: official LOZO (`third_party/LOZO/large_models/run_lozo.py`)
- vLLM path: shared LoZO-vLLM task runner
  (`python -m zo_vllm.experiment.runners.vllm_zo_task`)
- Default task: SST-2, official `{train=1000, dev=500, eval=872}` split sampling
- Task management: HF `datasets.Dataset` splits are owned by `zo_vllm.tasks`
  adapters; Phase 4 runners only select a task and orchestrate backends.
- Seeds: `seed=42` for the official LOZO/Trainer random stream and
  `task.data_seed=0` for SST-2 train/dev/eval sampling
- Default OPT paper-grid profile is copied from LOZO/MeZO scripts and paper Table 5:
  - `steps=20000`
  - `eval_interval=4000`
  - `batch_size=16`
  - `lr={1e-6,1e-7}`
  - `eps={1e-3,1e-4}`
  - `rank={1,2,4}`
  - `nu={50,100}`
  - `seed=0`

## Directory Layout

```text
phase4/
  configs/
  runners/
  results/
  README.md
```

All checkpoints, logs, manifests, summaries, and run outputs stay under
`phase4/`.

## Run Plan

The OPT-13B alignment configs are:

```text
phase4/configs/phase4_sst2_opt13b_official_lozo_paper_grid.json
phase4/configs/phase4_sst2_opt13b_vllm_paper_grid.json
```

They include:

- `lozo` official baseline, launched through the unmodified LOZO repository
- `vllm` optimized path, using the same SST-2 classification loss/accuracy口径

The official LOZO backend intentionally launches the unmodified LOZO repository
through the shared `zo_vllm.experiment.runners.backend_job` wrapper; Phase 4
does not depend on Phase 3 timing wrappers.

## Launch

```bash
.venv/bin/python -u phase4/runners/launch_phase4.py \
  --config phase4/configs/phase4_sst2_opt1p3b_long.json \
  --run-id phase4_opt1p3b_sst2_long_<timestamp> \
  --gpus 6,7
```

This launcher:

- starts one tmux window per job
- supports `--resume-existing` (skip completed, rerun failed/partial)
- writes per-job `run_state.json`, `manifest.json`, logs, and artifacts

## Collect Summary

```bash
.venv/bin/python -u phase4/runners/collect_phase4_summary.py \
  phase4/results/<run_id> \
  --output phase4/results/<run_id>/summary.md
```

Summary includes per-run config, final eval metric/loss, wall clock, step time,
throughput, convergence flag, resume flag, wandb links, and baseline-vs-vLLM
comparison notes.

## Task Configuration

Phase 4 configs use an HF-like nested task block:

```json
{
  "task": {
    "name": "sst2",
    "num_train": 1000,
    "num_dev": 500,
    "num_eval": 872,
    "data_seed": 0,
    "template": "default",
    "max_length": 2048,
    "max_new_tokens": 50
  }
}
```

Supported task adapters currently map to these backend objectives:

| task | official LOZO task | vLLM objective |
| --- | --- | --- |
| `sst2` | `SST2` | `sst2_classification` |
| `boolq` | `BoolQ` | `boolq_classification` |
| `squad` | `SQuAD` | `squad_nll` |

## Current OPT-13B Snapshot

The 2026-05-25 OPT-13B rank-2 convergence snapshot is summarized in:

```text
phase4/reports/opt13b_r2_lr1e7_eps1e3_nu50nu100_20260525_080736/summary.md
```

This report records completed vLLM `nu=50` and `nu=100` long runs plus the
currently available official LOZO baseline points. It includes step-based and
estimated wall-clock plots. The estimated wall-clock plots assume uniform
measured step time per run; use them for convergence visualization, not exact
per-metric timestamp claims.
