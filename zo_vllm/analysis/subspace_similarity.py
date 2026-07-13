"""Rank-aware AGZO subspace similarity metrics."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

import torch


VMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class PairSummary:
    """Summary statistics for a set of subspace-pair similarities."""

    tag: str
    num_pairs: int
    mean_weighted_subspace_similarity: float
    min_weighted_subspace_similarity: float
    p25_weighted_subspace_similarity: float
    median_weighted_subspace_similarity: float
    p75_weighted_subspace_similarity: float
    max_weighted_subspace_similarity: float
    pairs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "num_pairs": self.num_pairs,
            "mean_weighted_subspace_similarity": self.mean_weighted_subspace_similarity,
            "min_weighted_subspace_similarity": self.min_weighted_subspace_similarity,
            "p25_weighted_subspace_similarity": self.p25_weighted_subspace_similarity,
            "median_weighted_subspace_similarity": self.median_weighted_subspace_similarity,
            "p75_weighted_subspace_similarity": self.p75_weighted_subspace_similarity,
            "max_weighted_subspace_similarity": self.max_weighted_subspace_similarity,
            "pairs": self.pairs,
        }


def _resolve_device(v_maps: Sequence[VMap], device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    if device != "auto":
        raise ValueError(f"unsupported device: {device}")
    for v_map in v_maps:
        for tensor in v_map.values():
            if tensor.is_cuda:
                return tensor.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_column_matrix(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    matrix = tensor.detach().to(device=device, dtype=torch.float32)
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(1)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 1D or 2D V tensor, got shape={tuple(matrix.shape)}")
    return matrix.contiguous()


def _orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def weighted_subspace_similarity(
    left: VMap,
    right: VMap,
    *,
    device: str = "auto",
    force_svd: bool = False,
) -> dict[str, float]:
    """Compute weighted subspace similarity for one pair of V maps.

    For rank 1, the exact principal cosine is simply ``abs(cosine)``. Setting
    ``force_svd=True`` still routes through the generic SVD path, which is useful
    for tests or audits but gives the same value up to floating-point noise.
    """

    target_device = _resolve_device([left, right], device)
    weights: list[int] = []
    similarities: list[float] = []
    projection_similarities: list[float] = []
    min_principal_cosines: list[float] = []

    for name in sorted(set(left) & set(right)):
        a = _as_column_matrix(left[name], target_device)
        b = _as_column_matrix(right[name], target_device)
        if a.numel() == 0 or b.numel() == 0:
            continue
        rank = min(int(a.shape[1]), int(b.shape[1]))
        if rank <= 0:
            continue
        if rank == 1 and not force_svd:
            qa = torch.nn.functional.normalize(a[:, :1], dim=0)
            qb = torch.nn.functional.normalize(b[:, :1], dim=0)
            singular_values = (qa * qb).sum(dim=0).abs().clamp(0.0, 1.0)
        else:
            qa = _orthonormal_columns(a)[:, :rank]
            qb = _orthonormal_columns(b)[:, :rank]
            overlap = qa.transpose(0, 1).matmul(qb)
            singular_values = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
        projection = singular_values.square().sum().div(rank)
        similarity = projection.sqrt()
        weights.append(int(a.numel()))
        projection_similarities.append(float(projection.item()))
        similarities.append(float(similarity.item()))
        min_principal_cosines.append(float(singular_values.min().item()))

    if not weights:
        raise ValueError("no comparable V subspaces")

    total = float(sum(weights))
    weighted_similarity = sum(w * value for w, value in zip(weights, similarities)) / total
    weighted_projection = (
        sum(w * value for w, value in zip(weights, projection_similarities)) / total
    )
    return {
        "weighted_subspace_similarity": float(weighted_similarity),
        "weighted_projection_similarity": float(weighted_projection),
        "mean_subspace_similarity": float(sum(similarities) / len(similarities)),
        "min_subspace_similarity": float(min(similarities)),
        "max_subspace_similarity": float(max(similarities)),
        "mean_min_principal_cosine": float(
            sum(min_principal_cosines) / len(min_principal_cosines)
        ),
        "num_layers": int(len(similarities)),
        "total_v_elements": int(sum(weights)),
    }


def select_subspace_pairs(
    num_records: int,
    *,
    mode: str,
    random_pairs: int,
    seed: int,
    gaps: Sequence[int] = (2, 4, 8, 16),
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Select record-index pairs using the standard analysis modes."""

    if int(num_records) < 2:
        raise ValueError("num_records must be at least 2")
    if mode == "all_nonoverlap":
        return [
            (
                "all_nonoverlap",
                [(i, j) for i in range(num_records) for j in range(i + 1, num_records)],
            )
        ]
    if mode == "random_nonoverlap":
        all_pairs = [(i, j) for i in range(num_records) for j in range(i + 1, num_records)]
        if len(all_pairs) > int(random_pairs):
            rng = random.Random(int(seed))
            all_pairs = rng.sample(all_pairs, int(random_pairs))
        return [("random_nonoverlap", all_pairs)]
    if mode != "default":
        raise ValueError(f"unsupported pair selection mode: {mode}")

    rng = random.Random(int(seed))
    random_selected: list[tuple[int, int]] = []
    while len(random_selected) < int(random_pairs):
        i = rng.randrange(num_records)
        j = rng.randrange(num_records)
        if i != j:
            random_selected.append((min(i, j), max(i, j)))
    selected = [
        ("adjacent", [(i, i + 1) for i in range(num_records - 1)]),
        ("random", random_selected),
    ]
    for gap in gaps:
        pairs = [(i, i + int(gap)) for i in range(num_records - int(gap))]
        if pairs:
            selected.append((f"gap_{int(gap)}", pairs))
    return selected


def summarize_subspace_pairs(
    v_maps: Sequence[VMap],
    pairs: Sequence[tuple[int, int]],
    *,
    tag: str,
    device: str = "auto",
    batched: bool = True,
    include_pair_details: bool = True,
    force_svd: bool = False,
) -> PairSummary:
    """Summarize weighted subspace similarities for many record pairs."""

    if not pairs:
        raise ValueError("pairs must be non-empty")
    if batched:
        details = _batched_pair_metrics(v_maps, pairs, device=device, force_svd=force_svd)
    else:
        details = [
            {
                "left": int(i),
                "right": int(j),
                **weighted_subspace_similarity(
                    v_maps[i],
                    v_maps[j],
                    device=device,
                    force_svd=force_svd,
                ),
            }
            for i, j in pairs
        ]
    values = [float(item["weighted_subspace_similarity"]) for item in details]
    values_sorted = sorted(values)
    pair_details = details if include_pair_details else []
    return PairSummary(
        tag=tag,
        num_pairs=len(values),
        mean_weighted_subspace_similarity=float(sum(values) / len(values)),
        min_weighted_subspace_similarity=float(values_sorted[0]),
        p25_weighted_subspace_similarity=float(values_sorted[len(values_sorted) // 4]),
        median_weighted_subspace_similarity=float(values_sorted[len(values_sorted) // 2]),
        p75_weighted_subspace_similarity=float(
            values_sorted[(3 * len(values_sorted)) // 4]
        ),
        max_weighted_subspace_similarity=float(values_sorted[-1]),
        pairs=pair_details,
    )


def _batched_pair_metrics(
    v_maps: Sequence[VMap],
    pairs: Sequence[tuple[int, int]],
    *,
    device: str,
    force_svd: bool,
) -> list[dict[str, Any]]:
    target_device = _resolve_device(v_maps, device)
    pair_count = len(pairs)
    weighted_sum = torch.zeros(pair_count, device=target_device, dtype=torch.float64)
    weighted_projection_sum = torch.zeros(
        pair_count, device=target_device, dtype=torch.float64
    )
    layer_sum = torch.zeros(pair_count, device=target_device, dtype=torch.float64)
    min_layer = torch.full(
        (pair_count,), float("inf"), device=target_device, dtype=torch.float64
    )
    max_layer = torch.full(
        (pair_count,), float("-inf"), device=target_device, dtype=torch.float64
    )
    min_principal_sum = torch.zeros(pair_count, device=target_device, dtype=torch.float64)
    total_weight = torch.zeros(pair_count, device=target_device, dtype=torch.float64)
    layer_count = torch.zeros(pair_count, device=target_device, dtype=torch.int64)

    common_names = sorted(
        {
            name
            for left_idx, right_idx in pairs
            for name in (set(v_maps[left_idx]) & set(v_maps[right_idx]))
        }
    )
    if not common_names:
        raise ValueError("no common V tensors across records")

    for name in common_names:
        left_tensors = []
        right_tensors = []
        valid_indices = []
        for pair_index, (left_idx, right_idx) in enumerate(pairs):
            if name not in v_maps[left_idx] or name not in v_maps[right_idx]:
                continue
            left = _as_column_matrix(v_maps[left_idx][name], target_device)
            right = _as_column_matrix(v_maps[right_idx][name], target_device)
            if left.numel() == 0 or right.numel() == 0:
                continue
            if left.shape != right.shape:
                raise ValueError(
                    f"V shape mismatch for {name}: {tuple(left.shape)} vs {tuple(right.shape)}"
                )
            left_tensors.append(left)
            right_tensors.append(right)
            valid_indices.append(pair_index)
        if not valid_indices:
            continue

        left_batch = torch.stack(left_tensors, dim=0)
        right_batch = torch.stack(right_tensors, dim=0)
        rank = min(int(left_batch.shape[-1]), int(right_batch.shape[-1]))
        if rank <= 0:
            continue

        if rank == 1 and not force_svd:
            qa = torch.nn.functional.normalize(left_batch[..., :1], dim=1)
            qb = torch.nn.functional.normalize(right_batch[..., :1], dim=1)
            singular_values = (qa * qb).sum(dim=1).abs().clamp(0.0, 1.0)
        else:
            qa, _ = torch.linalg.qr(left_batch, mode="reduced")
            qb, _ = torch.linalg.qr(right_batch, mode="reduced")
            overlap = qa[..., :rank].transpose(-2, -1).matmul(qb[..., :rank])
            singular_values = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)

        projection = singular_values.square().sum(dim=-1).div(rank).to(torch.float64)
        similarity = projection.sqrt()
        min_principal = singular_values.min(dim=-1).values.to(torch.float64)
        weight = float(left_batch[0].numel())
        idx = torch.tensor(valid_indices, device=target_device, dtype=torch.long)
        weighted_sum[idx] += similarity * weight
        weighted_projection_sum[idx] += projection * weight
        layer_sum[idx] += similarity
        min_layer[idx] = torch.minimum(min_layer[idx], similarity)
        max_layer[idx] = torch.maximum(max_layer[idx], similarity)
        min_principal_sum[idx] += min_principal
        total_weight[idx] += weight
        layer_count[idx] += 1

    if bool((layer_count == 0).any().item()):
        raise ValueError("at least one pair has no comparable V subspaces")

    weighted_similarity = (weighted_sum / total_weight).detach().cpu().tolist()
    weighted_projection = (weighted_projection_sum / total_weight).detach().cpu().tolist()
    mean_layer = (layer_sum / layer_count.to(torch.float64)).detach().cpu().tolist()
    min_layer_list = min_layer.detach().cpu().tolist()
    max_layer_list = max_layer.detach().cpu().tolist()
    mean_min_principal = (
        min_principal_sum / layer_count.to(torch.float64)
    ).detach().cpu().tolist()
    layer_count_list = layer_count.detach().cpu().tolist()
    total_weight_list = total_weight.detach().cpu().tolist()

    details = []
    for pair_index, (left_idx, right_idx) in enumerate(pairs):
        details.append(
            {
                "left": int(left_idx),
                "right": int(right_idx),
                "weighted_subspace_similarity": float(weighted_similarity[pair_index]),
                "weighted_projection_similarity": float(weighted_projection[pair_index]),
                "mean_subspace_similarity": float(mean_layer[pair_index]),
                "min_subspace_similarity": float(min_layer_list[pair_index]),
                "max_subspace_similarity": float(max_layer_list[pair_index]),
                "mean_min_principal_cosine": float(mean_min_principal[pair_index]),
                "num_layers": int(layer_count_list[pair_index]),
                "total_v_elements": int(total_weight_list[pair_index]),
            }
        )
    return details
