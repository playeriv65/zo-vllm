# Third-Party Components

This directory contains external research/runtime code used by the experiments.

| component | path | use |
|---|---|---|
| vLLM-ZO | `third_party/vllm/` | vLLM runtime modified for LoZO-vLLM experiments |
| LOZO | `third_party/LOZO/large_models/run_lozo.py` | official LOZO baseline |
| MeZO | `third_party/LOZO/large_models/run_mezo.py` | official full-Gaussian MeZO baseline |

The MeZO baseline is kept through the LOZO repository because that repository
vendors the original `run_mezo.py` and `trainer.py` full-rank Gaussian ZO path.
