"""Console logging helpers for vLLM ZO runners."""

from __future__ import annotations

import os
from typing import Any


def print_vllm_run_header(
    *,
    args: Any,
    runtime_modes: Any,
    model_name: str,
    prequant_model: str | None,
    data_seed: int,
    output_paths: Any,
    step_nvtx_enabled: bool,
) -> None:
    """Print the main run configuration summary."""

    print(
        f"[vLLM] steps={args.steps} warmup_steps={args.warmup_steps} "
        f"batch_size={args.batch_size} rank={args.rank} "
        f"lr={args.lr} eps={args.eps} "
        f"estimator={args.estimator} "
        f"num_queries={args.num_queries} "
        f"perturbation_sides={args.perturbation_sides} "
        f"query_microbatch_size={args.query_microbatch_size} "
        f"direction_scale={runtime_modes.effective_direction_scale} "
        f"direction_scale_requested={args.direction_scale} "
        f"direction_scale_mode={runtime_modes.direction_scale_mode} "
        f"direction_scale_note='{runtime_modes.direction_scale_note}' "
        f"direction_scale_applies_to={runtime_modes.direction_scale_applies_to} "
        f"perturbation_normalization={args.perturbation_normalization} "
        f"profile_mode={args.profile_mode} "
        f"enforce_eager={args.enforce_eager} scoring_backend={args.scoring_backend} "
        f"sync_weight_update={args.sync_weight_update} "
        f"qkv_weight_update={args.qkv_weight_update} "
        f"direct_update_mode={args.direct_update_mode} "
        f"quantized_update_mode={args.quantized_update_mode} "
        f"update_bank_rank={args.update_bank_rank} "
        f"update_bank_rank_auto={args.update_bank_rank_auto} "
        f"update_bank_rank_requested={args.update_bank_rank_requested} "
        f"lora_slot_rank={runtime_modes.lora_slot_rank} "
        f"vllm_quantization={args.vllm_quantization} "
        f"prequant_model={prequant_model} "
        f"gradient_accumulation_update_steps={args.gradient_accumulation_update_steps} "
        f"clean_eval_mode=accumulated_lora_no_fold "
        f"u_beta={args.u_beta} "
        f"u_norm_cap={args.u_norm_cap} "
        f"u_snapshot_interval={args.u_snapshot_interval} "
        f"unified_stepper={int(runtime_modes.use_unified_stepper)} "
        f"direction_provider={args.direction_provider} "
        f"lozo_provider_mode={args.lozo_provider_mode} "
        f"nu={args.nu} agzo_kappa={args.agzo_kappa} "
        f"agzo_basis_method={args.agzo_basis_method} "
        f"u_dim={args.u_dim} "
        f"direct_worker_loss_impl={args.direct_worker_loss_impl} "
        f"score_chunk_size={args.score_chunk_size} "
        f"vllm_use_v2_model_runner={os.environ.get('VLLM_USE_V2_MODEL_RUNNER')} "
        f"direct_lora_from_directions={args.direct_lora_from_directions} "
        f"direction_sampling={args.direction_sampling} "
        f"train_objective={args.train_objective} "
        f"opt_bos_mode={args.opt_bos_mode} "
        f"step_nvtx={int(step_nvtx_enabled)} "
        f"model_name={model_name} seed={args.seed} data_seed={data_seed}",
        flush=True,
    )
    print(
        f"[vLLM] output_dir={output_paths.output_dir} "
        f"output_root={output_paths.output_root} "
        f"experiment_name={output_paths.experiment_name}",
        flush=True,
    )


def print_interval_summary(
    *,
    args: Any,
    batches_per_epoch: int,
    intervals: Any,
) -> None:
    """Print effective dataloader and interval settings."""

    print(
        f"[vLLM] dataloader_drop_last={int(bool(args.dataloader_drop_last))} "
        f"batches_per_epoch={batches_per_epoch} "
        f"eval_interval_steps={intervals.eval} "
        f"progress_interval_steps={intervals.progress} "
        f"train_loss_interval_steps={intervals.train_loss} "
        f"save_strategy={args.save_strategy} "
        f"save_steps={intervals.save}",
        flush=True,
    )
