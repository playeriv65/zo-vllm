"""Runtime-level native checkpoint payload operations."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from zo_vllm.core.weight_sync import (
    WeightSync,
    apply_embedding_lowrank_update_to_weight_,
    apply_lowrank_update_to_weight_,
)


DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3


def _apply_effective_direction_to_target_(
    target: torch.Tensor,
    *,
    hf_name: str,
    raw_direction: Mapping[str, torch.Tensor],
    precision: str,
) -> None:
    """Materialize one accumulated HF direction into a checkpoint tensor."""

    direction = dict(raw_direction)
    U = direction.get("U_accum")
    if U is None:
        U = direction["U"]
    V = direction["V"]
    apply_update = (
        apply_embedding_lowrank_update_to_weight_
        if "embed_tokens" in hf_name or hf_name == "lm_head.weight"
        else apply_lowrank_update_to_weight_
    )
    apply_update(
        target,
        U.to(device="cpu", non_blocking=False),
        V.to(device="cpu", non_blocking=False),
        c=-1.0,
        lr=1.0,
        weight_decay=0.0,
        precision=precision,
        direction_scale=1.0,
    )


def normalize_native_checkpoint_key(key: str) -> str:
    """Map runtime wrapper state keys back to base-model state keys."""

    return key.replace(".linear_method.base_layer.", ".").replace(".base_layer.", ".")


def normalize_native_checkpoint_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove runtime-only wrapper names from a native checkpoint state dict."""

    normalized: dict[str, torch.Tensor] = {}
    source_keys: dict[str, str] = {}
    for source_key, tensor in state_dict.items():
        target_key = normalize_native_checkpoint_key(source_key)
        previous = normalized.get(target_key)
        if previous is not None:
            same_storage = (
                previous.device == tensor.device
                and previous.data_ptr() == tensor.data_ptr()
            )
            if not same_storage:
                raise RuntimeError(
                    "native checkpoint key normalization collision: "
                    f"{source_keys[target_key]!r} and {source_key!r} both map to "
                    f"{target_key!r}"
                )
            continue
        normalized[target_key] = tensor
        source_keys[target_key] = source_key
    return normalized


def save_sharded_state(llm: Any, checkpoint_dir: str) -> None:
    """Save base-model-compatible shards without runtime wrapper key names."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = str(checkpoint_dir)

    def save_on_worker(worker):
        from safetensors.torch import save_file
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.model_loader import ShardedStateLoader

        model = worker.model_runner.model
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict = normalize_native_checkpoint_state(state_dict)
        rank = get_tensor_model_parallel_rank()
        result = _save_tensor_parts(
            state_dict,
            checkpoint_path=checkpoint_path,
            rank=int(rank),
            save_file=save_file,
        )
        return {"rank": int(rank), **result}

    llm.collective_rpc(save_on_worker)


def save_effective_native_checkpoint(
    *,
    llm: Any,
    checkpoint_dir: str,
    accumulated_update_state: Any | None,
    weight_sync: WeightSync,
    step: int,
    use_lora_bank_update: bool,
    precision: str,
) -> dict[str, Any]:
    """Save the effective model as a vLLM native sharded checkpoint."""

    fold_s = 0.0
    temporary_delta_s = 0.0
    if use_lora_bank_update:
        if accumulated_update_state is not None:
            flush = getattr(
                accumulated_update_state, "flush_pending_to_accumulated", None
            )
            if callable(flush):
                fold_s += float(flush(step=int(step)))
            directions = accumulated_update_state.clean_directions_for_score()
        else:
            directions = {}
        t0 = time.perf_counter()
        if directions:
            save_results = _save_lora_bank_effective_state_without_mutation(
                llm,
                checkpoint_dir=checkpoint_dir,
                weight_sync=weight_sync,
                directions=directions,
                precision=precision,
            )
        else:
            save_sharded_state(llm, checkpoint_dir)
            save_results = []
        temporary_delta_s = time.perf_counter() - t0
        return {
            "fold_s": fold_s,
            "temporary_delta_s": temporary_delta_s,
            "folded_accumulated_update": False,
            "temporary_effective_delta": False,
            "materialized_effective_delta": bool(directions),
            "save_results": save_results,
        }

    if accumulated_update_state is not None:
        fold = getattr(accumulated_update_state, "fold", None)
        if not callable(fold):
            raise RuntimeError(
                "accumulated update state does not expose fold(); cannot save "
                "an effective native checkpoint"
            )
        fold_s = float(fold(step=int(step)))
    save_sharded_state(llm, checkpoint_dir)
    return {
        "fold_s": fold_s,
        "temporary_delta_s": temporary_delta_s,
        "folded_accumulated_update": accumulated_update_state is not None,
        "temporary_effective_delta": False,
    }


def load_native_checkpoint_into_workers(llm: Any, checkpoint_dir: str) -> list[Any]:
    """Load a vLLM native sharded checkpoint into the live workers."""

    checkpoint_path = str(checkpoint_dir)

    def live_state_and_paths(worker):
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.model_loader import ShardedStateLoader

        model = worker.model_runner.model
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict = normalize_native_checkpoint_state(state_dict)
        rank = get_tensor_model_parallel_rank()
        rank_pattern = f"model-rank-{rank}-part-*.safetensors"
        paths = sorted(Path(checkpoint_path).glob(rank_pattern))
        if not paths:
            raise FileNotFoundError(
                f"native checkpoint has no shards for tensor-parallel rank {rank}: "
                f"{checkpoint_path}"
            )
        return state_dict, paths

    def validate_on_worker(worker):
        from safetensors import safe_open

        state_dict, paths = live_state_and_paths(worker)
        checkpoint_shapes: dict[str, tuple[int, ...]] = {}
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key in checkpoint_shapes:
                        raise RuntimeError(
                            f"duplicate tensor {key!r} across native checkpoint shards"
                        )
                    checkpoint_shapes[key] = tuple(shard.get_slice(key).get_shape())
        expected_keys = set(state_dict)
        checkpoint_keys = set(checkpoint_shapes)
        missing = sorted(expected_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - expected_keys)
        shape_mismatches = sorted(
            key
            for key in expected_keys & checkpoint_keys
            if tuple(state_dict[key].shape) != checkpoint_shapes[key]
        )
        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "native checkpoint does not exactly match the live model: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"shape_mismatches={shape_mismatches[:5]}"
            )
        return {
            "validated_tensors": int(len(state_dict)),
            "num_parts": int(len(paths)),
        }

    def load_on_worker(worker):
        from safetensors.torch import load_file

        state_dict, paths = live_state_and_paths(worker)
        loaded = 0
        for path in paths:
            shard = load_file(str(path), device="cpu")
            for key, tensor in shard.items():
                target = state_dict.get(key)
                if target is None or tuple(target.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        "native checkpoint changed after validation: "
                        f"incompatible tensor {key!r}"
                    )
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded += 1
        if loaded != len(state_dict):
            raise RuntimeError(
                "native checkpoint changed after validation: "
                f"loaded {loaded} of {len(state_dict)} tensors"
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "loaded_tensors": int(loaded),
            "num_parts": int(len(paths)),
        }

    llm.collective_rpc(validate_on_worker)
    return llm.collective_rpc(load_on_worker)


def clear_update_state_for_loaded_checkpoint(update_state: Any | None) -> None:
    if update_state is None:
        return
    for name in (
        "accumulated_u",
        "pending_u",
        "v_cache",
        "vt_cache",
        "bank_a",
        "zero_u",
        "current_blocks",
        "used_rank",
    ):
        value = getattr(update_state, name, None)
        if hasattr(value, "clear"):
            value.clear()


def _save_lora_bank_effective_state_without_mutation(
    llm: Any,
    *,
    checkpoint_dir: str,
    weight_sync: WeightSync,
    directions: Mapping[str, Mapping[str, torch.Tensor]],
    precision: str,
) -> list[Any]:
    hf_to_vllm_mapping = dict(weight_sync.hf_to_vllm_mapping)
    hf_to_slice = dict(weight_sync.hf_to_slice)
    checkpoint_path = str(checkpoint_dir)

    def save_on_worker(worker):
        from safetensors.torch import save_file
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.model_loader import ShardedStateLoader

        model = worker.model_runner.model
        os.makedirs(checkpoint_path, exist_ok=True)
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())

        def resolve_state_key(vllm_name: str) -> str:
            if vllm_name in state_dict:
                return vllm_name
            module_path = vllm_name.replace(".weight", "")
            candidates = (
                f"{module_path}.base_layer.weight",
                f"{module_path}.linear_method.base_layer.weight",
            )
            for candidate in candidates:
                if candidate in state_dict:
                    return candidate
            try:
                module = model.get_submodule(module_path)
                base_layer = getattr(module, "base_layer", module)
                weight = getattr(base_layer, "weight", None)
            except AttributeError:
                weight = None
            if weight is not None:
                target_ptr = weight.data.data_ptr()
                for key, tensor in state_dict.items():
                    if tensor.data_ptr() == target_ptr:
                        return key
            raise RuntimeError(
                "effective native checkpoint save could not find state_dict "
                f"tensor {vllm_name!r}"
            )

        directions_by_state_key: dict[
            str,
            list[tuple[str, str, Mapping[str, torch.Tensor]]],
        ] = {}
        for hf_name, raw_direction in directions.items():
            vllm_name = hf_to_vllm_mapping.get(hf_name)
            if vllm_name is None:
                raise ValueError(f"No mapping for HF parameter: {hf_name}")
            state_key = resolve_state_key(vllm_name)
            directions_by_state_key.setdefault(state_key, []).append(
                (hf_name, vllm_name, raw_direction)
            )

        rank = get_tensor_model_parallel_rank()
        materialized_state: dict[str, torch.Tensor] = {}
        num_materialized_tensors = 0

        for key, tensor in state_dict.items():
            tensor_cpu = tensor.detach().to(device="cpu", copy=True)
            key_directions = directions_by_state_key.get(key, ())
            if key_directions:
                num_materialized_tensors += 1
            for hf_name, vllm_name, raw_direction in key_directions:
                target = tensor_cpu
                if hf_name in hf_to_slice:
                    start, end = hf_to_slice[hf_name]
                    target = target[start:end, :]
                elif "qkv_proj.weight" in vllm_name:
                    hidden_size = target.shape[0] // 3
                    if ".q_proj.weight" in hf_name:
                        target = target[0:hidden_size, :]
                    elif ".k_proj.weight" in hf_name:
                        target = target[hidden_size : 2 * hidden_size, :]
                    elif ".v_proj.weight" in hf_name:
                        target = target[2 * hidden_size : 3 * hidden_size, :]
                    else:
                        raise ValueError(f"Unknown packed qkv projection: {hf_name}")
                _apply_effective_direction_to_target_(
                    target,
                    hf_name=hf_name,
                    raw_direction=raw_direction,
                    precision=precision,
                )
            checkpoint_key = normalize_native_checkpoint_key(key)
            if checkpoint_key in materialized_state:
                raise RuntimeError(
                    f"native checkpoint key normalization collision: {checkpoint_key!r}"
                )
            materialized_state[checkpoint_key] = tensor_cpu
        save_result = _save_tensor_parts(
            materialized_state,
            checkpoint_path=checkpoint_path,
            rank=int(rank),
            save_file=save_file,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "rank": int(rank),
            "num_directions": int(len(directions)),
            "num_materialized_tensors": int(num_materialized_tensors),
            **save_result,
        }

    return llm.collective_rpc(save_on_worker)


def _save_tensor_parts(
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: str,
    rank: int,
    save_file: Any,
) -> dict[str, int]:
    max_shard_bytes = int(
        os.environ.get(
            "VLLM_ZO_CHECKPOINT_MAX_CPU_SHARD_BYTES",
            str(DEFAULT_MAX_SHARD_BYTES),
        )
    )
    if max_shard_bytes <= 0:
        raise ValueError("VLLM_ZO_CHECKPOINT_MAX_CPU_SHARD_BYTES must be positive")
    part: dict[str, torch.Tensor] = {}
    part_bytes = 0
    part_index = 0

    def flush() -> None:
        nonlocal part, part_bytes, part_index
        if not part:
            return
        filename = f"model-rank-{rank}-part-{part_index}.safetensors"
        save_file(part, os.path.join(checkpoint_path, filename))
        part = {}
        part_bytes = 0
        part_index += 1

    for key, tensor in state_dict.items():
        tensor_bytes = tensor.nelement() * tensor.element_size()
        if part and part_bytes + tensor_bytes > max_shard_bytes:
            flush()
        part[key] = tensor
        part_bytes += tensor_bytes
    flush()
    return {
        "num_tensors": int(len(state_dict)),
        "num_parts": int(part_index),
        "max_cpu_shard_bytes": int(max_shard_bytes),
    }
