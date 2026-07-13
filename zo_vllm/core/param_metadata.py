"""Parameter metadata shared by direction and update backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @property
    def ndim(self) -> int:
        return len(self.shape)
