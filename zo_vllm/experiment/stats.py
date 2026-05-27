from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_tail(values: Sequence[float], tail_count: int) -> dict[str, float]:
    if tail_count <= 0:
        return summarize(values)
    return summarize(values[-tail_count:])
