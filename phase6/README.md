# Phase 6: MeZO Alignment with High-Rank Factorized ZO

Phase 6 tests whether high-rank factorized perturbations recover MeZO-like
optimization behavior while staying compatible with the vLLM/LoRA fast path.

## Mathematical Target

MeZO samples full-rank Gaussian perturbations:

```text
Z_ij ~ N(0, 1)
```

Phase 6 uses high-rank factorized perturbations:

```text
Z = U V^T / sqrt(r)
U_ik ~ N(0, 1), V_jk ~ N(0, 1)
```

This matches the marginal variance of full Gaussian perturbations as rank grows,
while preserving LoRA-compatible factorized execution. It is not exactly an iid
full Gaussian matrix because entries remain correlated through shared U/V
factors.

## Default Low-Cost Setting

```text
model=facebook/opt-1.3b
task=SST2
steps=1000
batch_size=16
num_train=1000
num_dev=500
num_eval=872
seed=42
train_set_seed=0
lr=1e-7
eps=1e-3
```

## Comparison Groups

| method | perturbation | execution path | status |
|---|---|---|---|
| MeZO baseline | full Gaussian Z | PyTorch / LOZO `run_mezo.py` | complete |
| Factorized-ZO r=128 | `UV^T/sqrt(r)` | vLLM | complete |
| Factorized-ZO r=256 | `UV^T/sqrt(r)` | vLLM | complete |
| Factorized-ZO r=512 | `UV^T/sqrt(r)` | vLLM | complete |

## Implementation Notes

The vLLM runner uses:

```text
--rank <r>
--direction-scale <1/sqrt(r)>
```

`--direction-scale` affects both plus/minus LoRA perturbations and base-weight
updates. Existing Phase 3/4 runs keep the default `direction_scale=1.0`.

## Metrics

Record:

- train loss
- eval loss and eval accuracy
- final and best accuracy
- step time
- wall-clock time
- speedup versus MeZO baseline
- optional `loss_plus`, `loss_minus`, and directional derivative `c`

## Results

All listed runs use the default low-cost setting above. The comparison keeps
runs with a clearly descending eval-loss trajectory; no flat/no-drop run is
used as alignment evidence.

The official MeZO path first reports intermediate eval loss at step 200, so
the headline loss-change comparison uses the common `step 200 -> step 1000`
window.

| method | eval loss @200 | eval loss @1000 | loss change 200->1000 | final dev acc | final valid acc | runtime s | speedup vs MeZO |
|---|---:|---:|---:|---:|---:|---:|---:|
| MeZO baseline | 0.833008 | 0.719238 | -0.113770 | 0.626 | 0.569954 | 79.56 | - |
| Factorized-ZO r=128 | 0.835074 | 0.725280 | -0.109794 | 0.632 | 0.589844 | 31.16 | 2.55x |
| Factorized-ZO r=256 | 0.837088 | 0.727933 | -0.109155 | 0.630 | 0.589844 | 31.28 | 2.54x |
| Factorized-ZO r=512 | 0.830685 | 0.729974 | -0.100711 | 0.630 | 0.589844 | 35.59 | 2.24x |

Detailed curves and machine-readable results are in `phase6/results/summary.md`
and `phase6/results/summary.json`.

The official MeZO baseline emits intermediate eval loss during training and
final dev/validation accuracy after training. Intermediate eval accuracy is
therefore available for vLLM runs, but not for the unmodified MeZO baseline.
