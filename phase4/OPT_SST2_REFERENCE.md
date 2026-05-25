# OPT + SST-2 Reproduction Reference (LOZO/MeZO Paper Notes)

This note records reproduction-oriented references for OPT + SST-2 from the
LOZO/MeZO paper discussion. It is used as an internal checklist for Phase 4.

## 1) Main-result checkpoints to target

### OPT-13B on SST-2 (Table 2, 1000 examples)

| Method | SST-2 Acc |
|---|---:|
| Zero-shot | 58.8 |
| ICL | 87.0 |
| MeZO | 91.3 |
| MeZO-LoRA | 89.6 |
| LOZO | 91.7 |
| FT | 91.8 |

Key point: LOZO is +0.4 over MeZO and close to FT (-0.1).

### OPT-30B on SST-2 (Table 3)

| Method | SST-2 Acc |
|---|---:|
| Zero-shot | 56.7 |
| ICL | 81.9 |
| MeZO | 90.7 |
| LOZO | 92.8 |

Key point: LOZO is +2.1 over MeZO on OPT-30B.

## 2) Convergence trend references

### OPT-30B + SST-2 (Figure 3, middle panel)

Approximate visual trend (not exact numeric points):

| Method | Init loss | ~50 epoch | ~100 epoch | ~300 epoch |
|---|---:|---:|---:|---:|
| MeZO | ~0.85-0.90 | ~0.45 | ~0.38-0.40 | ~0.30 |
| LOZO | ~0.85-0.90 | ~0.40 | ~0.35 | ~0.28-0.30 |

Interpretation: LOZO curve is generally below MeZO and drops faster early.

## 3) OPT-1.3B ablation reference (Appendix C.3)

SST-2 table for rank `r` and lazy interval `nu`:

| r | nu | SST-2 Acc | train loss |
|---:|---:|---:|---:|
| 1 | 50 | 88.1 | 0.45 |
| 1 | 100 | 89.0 | 0.46 |
| 2 | 1 | 55.0 | 0.79 |
| 2 | 50 | 93.0 | 0.37 |
| 2 | 100 | 92.1 | 0.37 |
| 2 | 200 | 92.7 | 0.37 |
| 2 | 500 | 91.7 | 0.37 |
| 4 | 50 | 91.3 | 0.35 |
| 4 | 100 | 92.0 | 0.35 |
| 8 | 50 | 88.5 | 0.48 |
| 8 | 100 | 88.9 | 0.45 |

Critical warning: `nu=1` can be very poor for SST-2 (r=2, nu=1 gives 55.0).

### OPT-1.3B + SST-2 loss trend (Figure 4, right panel)

Compared `nu = 1, 50, 100, 200, 500`:

- `nu=1`: poor/unstable, final loss around 0.79
- `nu=50/100/200/500`: fast drop, final around 0.37

## 4) Hyperparameter search grids (Appendix C.2 / Table 5)

### LOZO

- batch size: `16`
- learning rate: `{1e-6, 1e-7}`
- epsilon: `{1e-3, 1e-4}`
- rank `r`: `{1, 2, 4}`
- interval `nu`: `{50, 100}`

### MeZO

- batch size: `16`
- learning rate: `{1e-6, 1e-7}` (extra `5e-7` for SQuAD/DROP only)
- epsilon: `1e-3`

### MeZO-LoRA

- batch size: `16`
- learning rate: `{1e-4, 5e-5}` (extra `1e-5` for SQuAD/DROP only)
- epsilon: `1e-2`
- LoRA `(r, alpha) = (8, 16)`

### FT

- batch size: `8`
- learning rate: `{1e-5, 5e-5, 8e-5}`

### Training length / checkpointing

- MeZO/LOZO on OPT: `20,000` steps
- FT / FT-LoRA: around `2,000` steps (about 1/10 of LOZO length)
- Save best checkpoint every 1/5 of total steps:
  - for OPT 20,000-step runs: every `4,000` steps

## 5) Implementation caveat for LOZO LR definition

Paper note: LOZO learning rate is described as `alpha / r`.

If implementation uses update
`X <- X - alpha * c * (U V^T / r)`,
verify whether CLI `lr` maps to paper `alpha/r` or to `alpha`.

## 6) Recommended reproduction priority

For quick correctness checks, start from OPT-1.3B + SST-2:

- model: `facebook/opt-1.3b`
- method: LOZO
- batch size: `16`
- steps: `20000`
- rank `r`: `2` first
- interval `nu`: `50` first, then `100`/`200`
- epsilon: `1e-3` first, then `1e-4`
- lr: try `1e-6` then `1e-7`
- save/eval interval: every `4000` steps
- practical target: accuracy around `92-93`, train loss around `0.37`

Then compare against MeZO baseline with:

- batch size `16`
- steps `20000`
- epsilon `1e-3`
- lr `{1e-6, 1e-7}`

## 7) Usage in this repository

This document is a reference note, not a replacement for direct paper tables.
Use it to define default Phase 4 experiment lanes and to sanity-check trends.
