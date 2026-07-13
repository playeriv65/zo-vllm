"""Generic AGZO subspace collection helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Mapping

import torch

if TYPE_CHECKING:
    from zo_vllm.engine import ZOVLLMEngine


DirectionMap = Mapping[str, Mapping[str, torch.Tensor]]


@dataclass(frozen=True)
class SubspaceCollectionConfig:
    """Configuration for collecting V subspaces before a ZO training run."""

    rank: int
    power_iter_steps: int = 5
    basis_method: str = "power_iter"
    low_rank_oversample: int = 4
    summary_device: str = "auto"
    activation_force_eager: bool = True
    basis_seed_offset: int = 100000
    perturb_seed_offset: int = 200000


@dataclass
class SubspaceRecord:
    """One collected V subspace and collection metadata."""

    index: int
    batch_index: int
    basis_seed: int
    perturb_seed: int
    kappa: int
    num_layers: int
    elapsed_s: float
    profile_s: dict[str, Any] = field(default_factory=dict)
    v: dict[str, torch.Tensor] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def event_payload(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "batch_index": int(self.batch_index),
            "basis_seed": int(self.basis_seed),
            "perturb_seed": int(self.perturb_seed),
            "kappa": int(self.kappa),
            "num_layers": int(self.num_layers),
            "elapsed_s": float(self.elapsed_s),
            "profile_s": dict(self.profile_s),
        }


def keep_v_only(
    directions: DirectionMap,
    *,
    summary_device: str = "auto",
) -> dict[str, torch.Tensor]:
    """Detach and retain only V tensors from a direction map."""

    result: dict[str, torch.Tensor] = {}
    for name, value in directions.items():
        tensor = value["V"].detach()
        if summary_device == "cpu":
            tensor = tensor.cpu()
        elif summary_device == "cuda":
            if not tensor.is_cuda:
                tensor = tensor.cuda()
        elif summary_device != "auto":
            raise ValueError(f"unsupported summary_device: {summary_device}")
        result[name] = tensor.contiguous()
    return result


def collect_v_subspaces(
    engine: "ZOVLLMEngine",
    token_id_group_batches_by_subspace: Sequence[Sequence[Sequence[Sequence[int]]]],
    *,
    config: SubspaceCollectionConfig,
    start_index: int = 0,
    stride: int = 1,
    event_writer: Any | None = None,
) -> list[SubspaceRecord]:
    """Collect AGZO V subspaces from caller-provided token-id batches.

    ``token_id_group_batches_by_subspace`` is organized as:
    ``subspace -> kappa chunk -> token-id group -> token ids``. Dataset loading,
    prompting, and task objectives stay outside this generic module.
    """

    if int(config.rank) <= 0:
        raise ValueError("rank must be positive")
    if int(config.power_iter_steps) <= 0:
        raise ValueError("power_iter_steps must be positive")
    if config.basis_method not in {"power_iter", "svd", "low_rank_svd"}:
        raise ValueError(f"unsupported basis_method: {config.basis_method}")

    records: list[SubspaceRecord] = []
    for offset, raw_batches in enumerate(token_id_group_batches_by_subspace):
        if not raw_batches:
            raise ValueError("each subspace must contain at least one token-id batch")
        index = int(start_index) + offset
        batch_index = index * int(stride)
        batches = [
            [list(token_ids) for token_ids in token_id_groups]
            for token_id_groups in raw_batches
        ]
        basis_seed = int(config.basis_seed_offset) + index
        perturb_seed = int(config.perturb_seed_offset) + index
        step_start = time.perf_counter()
        if len(batches) == 1:
            directions, raw = engine.collect_agzo_directions(
                batches[0],
                activation_force_eager=bool(config.activation_force_eager),
                agzo_rank=int(config.rank),
                agzo_power_iter_steps=int(config.power_iter_steps),
                agzo_low_rank_oversample=int(config.low_rank_oversample),
                agzo_basis_seed=basis_seed,
                agzo_perturb_seed=perturb_seed,
                agzo_basis_method=config.basis_method,
            )
        else:
            directions, raw = engine.collect_agzo_directions_chunked(
                batches,
                activation_force_eager=bool(config.activation_force_eager),
                agzo_rank=int(config.rank),
                agzo_power_iter_steps=int(config.power_iter_steps),
                agzo_low_rank_oversample=int(config.low_rank_oversample),
                agzo_basis_seed=basis_seed,
                agzo_perturb_seed=perturb_seed,
                agzo_basis_method=config.basis_method,
            )
        elapsed = time.perf_counter() - step_start
        v_only = keep_v_only(directions, summary_device=config.summary_device)
        record = SubspaceRecord(
            index=index,
            batch_index=batch_index,
            basis_seed=basis_seed,
            perturb_seed=perturb_seed,
            kappa=len(batches),
            num_layers=len(v_only),
            elapsed_s=elapsed,
            profile_s=dict(raw.get("profile_s", {})),
            v=v_only,
            raw=raw,
        )
        records.append(record)
        if event_writer is not None:
            event_writer({"event": "collect", **record.event_payload()})
    return records
