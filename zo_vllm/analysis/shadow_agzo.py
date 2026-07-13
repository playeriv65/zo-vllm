"""HF shadow helpers for checking worker-side AGZO directions."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def make_padded_token_batch(
    token_groups: list[list[int]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Build a padded HF input batch from token-id groups."""

    max_len = max(len(ids) for ids in token_groups)
    input_ids = []
    attention_mask = []
    for ids in token_groups:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def power_iteration(activation_2d: torch.Tensor, rank: int, num_steps: int) -> torch.Tensor:
    """Estimate top right singular vectors through power iteration."""

    hidden_dim = activation_2d.shape[1]
    q = torch.randn(hidden_dim, rank, device=activation_2d.device, dtype=torch.float32)
    q, _ = torch.linalg.qr(q, mode="reduced")
    activation = activation_2d.float()
    for _ in range(max(1, int(num_steps))):
        y = activation.matmul(q)
        z = activation.transpose(0, 1).matmul(y)
        q, _ = torch.linalg.qr(z, mode="reduced")
    basis = q.transpose(0, 1)
    return basis / (basis.norm(p=2, dim=1, keepdim=True) + 1e-12)


def activation_basis(
    activation_2d: torch.Tensor,
    *,
    rank: int,
    basis_method: str,
    power_iter_steps: int,
    low_rank_oversample: int,
) -> torch.Tensor:
    """Return row-major AGZO basis vectors for one activation matrix."""

    activation = activation_2d.float()
    if basis_method == "power_iter":
        return power_iteration(activation, rank, power_iter_steps)
    if basis_method == "svd":
        _u, _s, vh = torch.linalg.svd(activation, full_matrices=False)
        basis = vh[: int(rank)]
    elif basis_method == "low_rank_svd":
        sample_rank = min(
            int(activation.shape[1]),
            int(rank) + max(0, int(low_rank_oversample)),
        )
        _u, _s, v = torch.svd_lowrank(
            activation,
            q=max(int(rank), sample_rank),
            niter=max(1, int(power_iter_steps)),
        )
        basis = v[:, : int(rank)].transpose(0, 1)
    else:
        raise ValueError(f"unsupported basis_method: {basis_method}")
    return basis / (basis.norm(p=2, dim=1, keepdim=True) + 1e-12)


@torch.no_grad()
def collect_shadow_agzo_directions(
    model,
    batch: dict[str, torch.Tensor],
    *,
    rank: int,
    power_iter_steps: int,
    low_rank_oversample: int,
    basis_method: str,
    basis_seed: int,
    perturb_seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Collect AGZO directions through a local HF model for worker validation."""

    params = dict(model.named_parameters())
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(param_name: str):
        def hook(_module, inputs, _output):
            for value in inputs:
                if torch.is_tensor(value):
                    activations[param_name] = value.detach()
                    return

        return hook

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        param_name = f"{module_name}.weight"
        param = params.get(param_name)
        if param is None or param.dim() != 2:
            continue
        hooks.append(module.register_forward_hook(make_hook(param_name)))

    device = next(model.parameters()).device
    rng_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(basis_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(basis_seed))
        _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )

        attention_mask = batch["attention_mask"].to(device=device, dtype=torch.bool)
        basis_by_name: dict[str, torch.Tensor] = {}
        for name, activation in activations.items():
            param = params[name]
            max_rank = min(
                int(rank),
                int(param.shape[0]),
                int(param.shape[1]),
                int(activation.shape[-1]),
            )
            if (
                activation.ndim >= 3
                and activation.shape[0] == attention_mask.shape[0]
                and activation.shape[1] == attention_mask.shape[1]
            ):
                activation_2d = activation[attention_mask]
            elif activation.ndim == 2 and activation.shape[0] == attention_mask.numel():
                activation_2d = activation[attention_mask.reshape(-1)]
            else:
                activation_2d = activation.reshape(-1, activation.shape[-1])
            basis_by_name[name] = activation_basis(
                activation_2d,
                rank=max_rank,
                basis_method=basis_method,
                power_iter_steps=power_iter_steps,
                low_rank_oversample=low_rank_oversample,
            ).to(
                device=param.device,
                dtype=param.dtype,
            )

        torch.manual_seed(int(perturb_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(perturb_seed))
        directions: dict[str, dict[str, torch.Tensor]] = {}
        for name, basis in basis_by_name.items():
            param = params[name]
            max_rank = int(basis.shape[0])
            r = torch.normal(
                mean=0.0,
                std=1.0,
                size=(param.shape[0], max_rank),
                device=param.device,
                dtype=param.dtype,
            )
            directions[name] = {
                "U": (r / math.sqrt(max_rank)).contiguous(),
                "V": basis.transpose(0, 1).contiguous(),
            }
        return directions
    finally:
        for hook in hooks:
            hook.remove()
        torch.set_rng_state(rng_state)
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)


def compare_agzo_directions(
    shadow: dict[str, dict[str, torch.Tensor]],
    worker: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Compare two AGZO direction maps by shape, max error, and V cosine."""

    shadow_keys = set(shadow)
    worker_keys = set(worker)
    common = sorted(shadow_keys & worker_keys)
    max_u_abs = 0.0
    max_v_abs = 0.0
    min_v_cos = 1.0
    worst_u = None
    worst_v = None
    shape_mismatches = []
    for key in common:
        s_u = shadow[key]["U"].detach().float().cpu()
        w_u = worker[key]["U"].detach().float().cpu()
        s_v = shadow[key]["V"].detach().float().cpu()
        w_v = worker[key]["V"].detach().float().cpu()
        if tuple(s_u.shape) != tuple(w_u.shape) or tuple(s_v.shape) != tuple(w_v.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "shadow_u": list(s_u.shape),
                    "worker_u": list(w_u.shape),
                    "shadow_v": list(s_v.shape),
                    "worker_v": list(w_v.shape),
                }
            )
            continue
        u_abs = float((s_u - w_u).abs().max().item())
        v_abs = float((s_v - w_v).abs().max().item())
        v_cos = float(
            torch.nn.functional.cosine_similarity(
                s_v.flatten(),
                w_v.flatten(),
                dim=0,
            ).item()
        )
        if u_abs > max_u_abs:
            max_u_abs = u_abs
            worst_u = key
        if v_abs > max_v_abs:
            max_v_abs = v_abs
            worst_v = key
        min_v_cos = min(min_v_cos, v_cos)
    return {
        "shadow_num_keys": len(shadow_keys),
        "worker_num_keys": len(worker_keys),
        "common_num_keys": len(common),
        "missing_in_worker": sorted(shadow_keys - worker_keys)[:20],
        "extra_in_worker": sorted(worker_keys - shadow_keys)[:20],
        "shape_mismatches": shape_mismatches[:20],
        "max_u_abs": max_u_abs,
        "max_u_abs_key": worst_u,
        "max_v_abs": max_v_abs,
        "max_v_abs_key": worst_v,
        "min_v_cos": min_v_cos,
    }
