import os
from pathlib import Path


def hf_cache_env(project_root: str | Path) -> dict[str, str]:
    del project_root
    cache_root = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", cache_root / "hub"))
    return {
        "HF_HOME": str(cache_root),
        "HF_DATASETS_CACHE": os.environ.get(
            "HF_DATASETS_CACHE",
            str(cache_root / "datasets"),
        ),
        "HF_HUB_CACHE": str(hub_cache),
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", str(cache_root / "xet")),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", str(hub_cache)),
    }


def configure_hf_cache(project_root: str | Path) -> Path:
    for key, value in hf_cache_env(project_root).items():
        os.environ.setdefault(key, value)
    return Path(os.environ["HF_HOME"])


def configure_vllm_training_env() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("WANDB_MODE", "offline")
