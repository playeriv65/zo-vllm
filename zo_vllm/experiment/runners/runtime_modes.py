"""Resolved runtime-mode flags for the vLLM ZO task runner."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import math

from zo_vllm.core.lora_scope import (
    is_infinite_nu,
    resolve_update_bank_rank,
    validate_nu,
)
from zo_vllm.tasks.superglue.record import OBJECTIVE_NAME as RECORD_NLL_OBJECTIVE
from zo_vllm.training.objective_router import is_option_classification_objective


@dataclass(frozen=True)
class VLLMZORuntimeModes:
    direct_lora_from_directions: bool
    use_lora_bank_update: bool
    use_accumulated_update: bool
    use_accumulated_lora_eval: bool
    effective_direction_scale: float
    direction_scale_mode: str
    direction_scale_note: str
    direction_scale_applies_to: str
    use_unified_stepper: bool
    lora_slot_rank: int


def resolve_vllm_zo_runtime_modes(args: Namespace) -> VLLMZORuntimeModes:
    """Validate interdependent runner args and return derived runtime flags."""

    direct_lora_from_directions = bool(int(args.direct_lora_from_directions))
    use_lora_bank_update = args.quantized_update_mode == "lora_bank"
    use_accumulated_update = (
        args.direct_update_mode == "accumulate" and not use_lora_bank_update
    )
    use_accumulated_lora_eval = use_accumulated_update or use_lora_bank_update
    try:
        args.nu = validate_nu(args.nu)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if use_lora_bank_update:
        _resolve_lora_bank_rank(args)
    else:
        args.update_bank_rank_auto = False
        args.update_bank_rank_requested = args.update_bank_rank
    if use_accumulated_update:
        _validate_accumulated_update_args(args, direct_lora_from_directions)
    _validate_accumulated_lora_eval_args(args, use_accumulated_lora_eval)
    _validate_direction_provider_args(args)

    effective_direction_scale = float(args.direction_scale)
    direction_scale_mode = (
        "default_original" if effective_direction_scale == 1.0 else "explicit"
    )
    direction_scale_note = (
        "original LOZO/vLLM unnormalized scale"
        if effective_direction_scale == 1.0
        else "explicit user scale"
    )
    direction_scale_applies_to = (
        "plus_minus_probe,accumulated_update,base_weight_update,fused_agzo_worker"
    )
    use_unified_stepper = (
        (
            args.train_objective in {"prompt_nll", "squad_nll"}
            or is_option_classification_objective(args.train_objective)
            or args.train_objective == RECORD_NLL_OBJECTIVE
        )
        and args.scoring_backend == "direct_worker"
        and direct_lora_from_directions
    )
    if not use_unified_stepper:
        raise SystemExit(
            "zo_vllm.experiment.runners.vllm_zo_task now requires the unified "
            "VLLMZOModel/ZOStepper "
            "path. Use GPU direct LoRA, direct-worker scoring, and "
            "--direct-lora-from-directions 1."
        )
    lora_slot_rank = (
        int(args.update_bank_rank) if use_lora_bank_update else int(args.rank)
    )
    return VLLMZORuntimeModes(
        direct_lora_from_directions=direct_lora_from_directions,
        use_lora_bank_update=use_lora_bank_update,
        use_accumulated_update=use_accumulated_update,
        use_accumulated_lora_eval=use_accumulated_lora_eval,
        effective_direction_scale=effective_direction_scale,
        direction_scale_mode=direction_scale_mode,
        direction_scale_note=direction_scale_note,
        direction_scale_applies_to=direction_scale_applies_to,
        use_unified_stepper=use_unified_stepper,
        lora_slot_rank=lora_slot_rank,
    )


def _resolve_lora_bank_rank(args: Namespace) -> None:
    if args.direct_update_mode != "accumulate":
        raise SystemExit(
            "--quantized-update-mode lora_bank requires --direct-update-mode accumulate"
        )
    requested_update_bank_rank = args.update_bank_rank
    total_target_steps = int(args.steps) + int(args.warmup_steps)
    estimated_blocks = (
        1
        if is_infinite_nu(int(args.nu))
        else int(math.ceil(total_target_steps / max(1, int(args.nu))))
    )
    estimated_block_rank = int(args.rank)
    try:
        args.update_bank_rank, update_bank_rank_auto = resolve_update_bank_rank(
            requested_update_bank_rank,
            rank=int(args.rank),
            steps=int(args.steps),
            warmup_steps=int(args.warmup_steps),
            nu=int(args.nu),
            rank_multiplier=1,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.update_bank_rank_auto = bool(update_bank_rank_auto)
    args.update_bank_rank_requested = requested_update_bank_rank
    if int(args.update_bank_rank) < int(args.rank):
        raise SystemExit("--update-bank-rank must be greater than or equal to --rank")
    estimated_rank = estimated_blocks * estimated_block_rank
    if estimated_rank > int(args.update_bank_rank):
        raise SystemExit(
            "--update-bank-rank is too small for the configured run: "
            f"estimated need rank {estimated_rank} "
            f"({estimated_blocks} refresh blocks * effective rank {estimated_block_rank}), "
            f"capacity {int(args.update_bank_rank)}"
        )


def _validate_accumulated_update_args(
    args: Namespace,
    direct_lora_from_directions: bool,
) -> None:
    if args.weight_update != "direct":
        raise SystemExit("--direct-update-mode accumulate requires --weight-update direct")
    if args.weight_update_precision != "param":
        raise SystemExit(
            "--direct-update-mode accumulate requires --weight-update-precision param"
        )
    if not direct_lora_from_directions:
        raise SystemExit(
            "--direct-update-mode accumulate requires --direct-lora-from-directions 1"
        )


def _validate_accumulated_lora_eval_args(
    args: Namespace,
    use_accumulated_lora_eval: bool,
) -> None:
    if args.u_snapshot_interval > 0 and not use_accumulated_lora_eval:
        raise SystemExit(
            "--u-snapshot-interval requires accumulated or lora_bank update mode"
        )
    if float(args.u_beta) != 1.0 and not use_accumulated_lora_eval:
        raise SystemExit("--u-beta requires accumulated or lora_bank update mode")
    if args.u_norm_cap is not None and not use_accumulated_lora_eval:
        raise SystemExit("--u-norm-cap requires accumulated or lora_bank update mode")
    if int(args.gradient_accumulation_update_steps) < 0:
        raise SystemExit("--gradient-accumulation-update-steps must be non-negative")
    if (
        int(args.gradient_accumulation_update_steps) > 0
        and not use_accumulated_lora_eval
    ):
        raise SystemExit(
            "--gradient-accumulation-update-steps requires accumulated or lora_bank update mode"
        )
    if not (0.0 <= float(args.u_beta) <= 1.0):
        raise SystemExit("--u-beta must be in [0, 1]")
    if args.u_norm_cap is not None and float(args.u_norm_cap) <= 0.0:
        raise SystemExit("--u-norm-cap must be positive")


def _validate_direction_provider_args(args: Namespace) -> None:
    if args.direction_provider in {"agzo", "uagzo", "suagzo"}:
        if int(args.agzo_kappa) <= 0:
            raise SystemExit("--kappa/--agzo-kappa must be positive")
    if args.direction_provider in {"uagzo", "suagzo"}:
        if args.u_dim is None or int(args.u_dim) <= 0:
            raise SystemExit(
                f"--u-dim must be positive for --direction-provider {args.direction_provider}"
            )
        if args.direction_provider == "uagzo" and int(args.u_dim) < int(args.rank):
            raise SystemExit("--u-dim must be greater than or equal to --rank")
