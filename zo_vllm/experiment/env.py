import os
from pathlib import Path


def configure_hf_cache(project_root: str | Path) -> Path:
    cache_root = Path(project_root) / ".cache" / "hf"
    os.environ.setdefault("HF_HOME", str(cache_root / "home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_root / "xet"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    return cache_root


def configure_vllm_training_env() -> None:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("WANDB_MODE", "offline")
