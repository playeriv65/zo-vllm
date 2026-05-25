# Phase 4 OPT-13B SST-2 Snapshot

Run directory: `phase4/results/phase4_opt13b_r2_lr1e7_eps1e3_nu50nu100_20260525_080736`

Snapshot time: 2026-05-25 09:14 local server time for completed vLLM runs.

## Configuration

Common settings:

| field | value |
|---|---|
| model | `facebook/opt-13b` |
| task | `SST2` |
| train/dev/eval samples | `1000 / 500 / 872` |
| steps | `20000` |
| batch size | `16` |
| rank | `2` |
| learning rate | `1e-7` |
| eps | `1e-3` |
| seed | `42` |
| train_set_seed | `0` |
| eval interval | `4000` |
| logging steps | `10` |

Jobs:

| job | backend | scope | nu | status |
|---|---|---|---:|---|
| `lozo_full_opt13b_r2_nu50_lr1e7_eps1e3` | official LOZO | full | 50 | running |
| `lozo_loraonly_opt13b_r2_nu50_lr1e7_eps1e3` | official LOZO | lora_only | 50 | running |
| `vllm_loraonly_opt13b_r2_nu50_lr1e7_eps1e3_trainloss10` | vLLM | lora_only | 50 | completed |
| `vllm_loraonly_opt13b_r2_nu100_lr1e7_eps1e3_trainloss10` | vLLM | lora_only | 100 | completed |

## Clean Eval Results

Primary convergence metrics are clean eval loss and eval accuracy on the current
base parameters. The 10-step training loss is not a clean objective; it is the
official LOZO plus-perturbation probe loss `loss(theta + eps * direction)`.

| job | step 4000 loss | step 8000 loss | step 12000 loss | step 16000 loss | step 20000 loss | step 20000 eval acc | final full eval acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOZO full nu50 | 0.413086 | pending | pending | pending | pending | pending | pending |
| LOZO lora_only nu50 | 0.428467 | 0.354736 | pending | pending | pending | pending | pending |
| vLLM lora_only nu50 | 0.399084 | 0.321895 | 0.263713 | 0.243588 | 0.231142 | 0.908000 | 0.932339 |
| vLLM lora_only nu100 | 0.401440 | 0.300538 | 0.244500 | 0.216440 | 0.205974 | 0.926000 | 0.927752 |

At the shared 4000-step point, both vLLM runs have lower clean eval loss than
official LOZO lora_only and are close to or below official LOZO full.

## Speed Snapshot

Step time was stable enough to estimate wall-clock curves using each run's mean
step time. These estimates exclude the discarded first vLLM restart and should be
treated as convergence-plot approximations, not precise per-metric timestamps.

| job | mean step time | projected 20k time | speedup vs LOZO lora_only | speedup vs LOZO full |
|---|---:|---:|---:|---:|
| LOZO full nu50 | 0.8081 s | 269.4 min | 0.89x | 1.00x |
| LOZO lora_only nu50 | 0.7219 s | 240.6 min | 1.00x | 1.12x |
| vLLM lora_only nu50 | 0.1598 s | 53.3 min | 4.52x | 5.06x |
| vLLM lora_only nu100 | 0.1612 s | 53.7 min | 4.48x | 5.01x |

## Plot Artifacts

Step-based plots:

- `plots/phase4_train_loss.svg`
- `plots/phase4_eval_loss.svg`
- `plots/phase4_eval_acc.svg`

Estimated wall-clock plots:

- `plots/phase4_estimated_wallclock_train_loss.svg`
- `plots/phase4_estimated_wallclock_eval_loss.svg`
- `plots/phase4_estimated_wallclock_eval_acc.svg`

The corresponding CSV files are stored next to the SVGs.

## Notes

- vLLM completed both long runs successfully with `exit_code=0`.
- Official LOZO full and lora_only baselines are still running and should not be
  stopped unless final loss/accuracy is clearly bad or the run fails.
- Training loss is useful only as a noisy debug signal because it is measured at
  `theta + eps * direction`; clean eval loss and accuracy are the main
  convergence evidence.
