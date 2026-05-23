"""Small helpers for step-by-step LOZO direction identity checks."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


def _update_tensor(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    cpu_tensor = tensor.detach().contiguous().cpu()
    hasher.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
    hasher.update(str(cpu_tensor.dtype).encode("ascii"))
    hasher.update(cpu_tensor.numpy().tobytes())


def digest_named_uv(
    items: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
) -> str:
    """Hash named U/V tensors in deterministic parameter order."""
    hasher = hashlib.sha256()
    for name, u_tensor, v_tensor in items:
        hasher.update(name.encode("utf-8"))
        _update_tensor(hasher, u_tensor)
        _update_tensor(hasher, v_tensor)
    return hasher.hexdigest()

