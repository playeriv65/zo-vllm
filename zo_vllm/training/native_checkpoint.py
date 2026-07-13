"""Runtime-level native checkpoint payload operations."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from zo_vllm.core.weight_sync import WeightSync, apply_lowrank_update_to_weight_


def save_sharded_state(llm: Any, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_sharded_state_fn = getattr(llm.llm_engine, "save_sharded_state", None)
    if callable(save_sharded_state_fn):
        save_sharded_state_fn(checkpoint_dir)
        return
    model_executor = getattr(llm.llm_engine, "model_executor", None)
    executor_save = getattr(model_executor, "save_sharded_state", None)
    if callable(executor_save):
        executor_save(checkpoint_dir)
        return
    llm.collective_rpc(
        "save_sharded_state",
        kwargs={"path": checkpoint_dir, "pattern": None, "max_size": None},
    )


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

    def load_on_worker(worker):
        from safetensors.torch import load_file
        from vllm.distributed import get_tensor_model_parallel_rank

        model = worker.model_runner.model
        state_dict = model.state_dict()
        rank = get_tensor_model_parallel_rank()
        loaded = 0
        skipped = []
        rank_pattern = f"model-rank-{rank}-part-*.safetensors"
        paths = sorted(Path(checkpoint_path).glob(rank_pattern))
        if not paths:
            paths = sorted(Path(checkpoint_path).glob("*.safetensors"))
        for path in paths:
            shard = load_file(str(path), device="cpu")
            for key, tensor in shard.items():
                target = state_dict.get(key)
                if target is None:
                    skipped.append(key)
                    continue
                if tuple(target.shape) != tuple(tensor.shape):
                    skipped.append(key)
                    continue
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "loaded_tensors": int(loaded),
            "skipped_tensors": int(len(skipped)),
            "first_skipped": skipped[:5],
        }

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

        max_shard_bytes = int(
            os.environ.get(
                "VLLM_ZO_CHECKPOINT_MAX_CPU_SHARD_BYTES",
                str(4 * 1024**3),
            )
        )

        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        part_bytes = 0
        state_dict_part: dict[str, torch.Tensor] = {}
        num_materialized_tensors = 0

        def flush_part() -> None:
            nonlocal part_idx, part_bytes, state_dict_part
            if not state_dict_part:
                return
            filename = f"model-rank-{rank}-part-{part_idx}.safetensors"
            save_file(state_dict_part, os.path.join(checkpoint_path, filename))
            part_idx += 1
            part_bytes = 0
            state_dict_part = {}

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
                direction = dict(raw_direction)
                U = direction.get("U_accum")
                if U is None:
                    U = direction["U"]
                V = direction["V"]
                apply_lowrank_update_to_weight_(
                    target,
                    U.to(device="cpu", non_blocking=False),
                    V.to(device="cpu", non_blocking=False),
                    c=-1.0,
                    lr=1.0,
                    weight_decay=0.0,
                    precision=precision,
                    direction_scale=1.0,
                )
            tensor_bytes = tensor_cpu.nelement() * tensor_cpu.element_size()
            if (
                max_shard_bytes > 0
                and state_dict_part
                and part_bytes + tensor_bytes > max_shard_bytes
            ):
                flush_part()
            state_dict_part[key] = tensor_cpu
            part_bytes += tensor_bytes
        flush_part()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "rank": int(rank),
            "num_directions": int(len(directions)),
            "num_materialized_tensors": int(num_materialized_tensors),
            "num_parts": int(part_idx),
            "max_cpu_shard_bytes": int(max_shard_bytes),
        }

    return llm.collective_rpc(save_on_worker)
