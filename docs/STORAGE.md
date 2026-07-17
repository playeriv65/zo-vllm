# Shared Storage

ZO-vLLM keeps runtime and compilation state on local storage while allowing
large model and experiment artifacts to live on a mounted shared filesystem.

Use the standard Hugging Face variable for model repositories:

```bash
export HF_HUB_CACHE=/shared/cache/huggingface/hub
```

Use `ZO_ARTIFACT_ROOT` for repository-owned run artifacts:

```bash
export ZO_ARTIFACT_ROOT=/shared/artifacts
```

Relative runner output paths are resolved below
`$ZO_ARTIFACT_ROOT/zo-vllm`. Absolute output paths remain unchanged, matching
the Hugging Face `TrainingArguments.output_dir` contract. Checkpoints remain
children of the resolved run directory; there is no separate checkpoint-root
setting.

Keep virtual environments, uv build state, compiler caches, and vLLM/Torch
compile caches on local storage. A mounted shared artifact root is intended for
model weights, checkpoints, metrics, and other durable run outputs.
