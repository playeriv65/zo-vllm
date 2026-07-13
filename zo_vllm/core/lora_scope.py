"""Shared LoRA target-module scope helpers."""

from __future__ import annotations

from zo_vllm.config import DEFAULT_ZO_TARGET_MODULES

DEFAULT_TRANSFORMER_TARGET_MODULES = list(DEFAULT_ZO_TARGET_MODULES)
LORA_NORMAL_SCOPE = "lora_normal"
LORA_FULL_SCOPE = "lora_full"
LORA_TRAIN_SCOPE_CHOICES = (LORA_NORMAL_SCOPE, LORA_FULL_SCOPE)

VLLM_MAX_LORA_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 512, 1024)
INFINITE_NU = -1


def is_infinite_nu(nu: int) -> bool:
    """Return whether nu means reuse one basis for the full run."""

    return int(nu) == INFINITE_NU


def validate_nu(nu: int) -> int:
    """Normalize and validate basis-refresh interval semantics."""

    nu_i = int(nu)
    if nu_i == INFINITE_NU:
        return nu_i
    if nu_i <= 0:
        raise ValueError("nu must be positive or -1 for infinite reuse")
    return nu_i


def should_refresh_for_nu(*, step: int, nu: int) -> bool:
    """Return whether a one-indexed step should refresh the current V basis."""

    step_i = int(step)
    if step_i <= 0:
        raise ValueError("step must be positive")
    nu_i = validate_nu(nu)
    if is_infinite_nu(nu_i):
        return step_i == 1
    return (step_i - 1) % nu_i == 0


def parse_target_modules(
    value: str | list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    modules: list[str] = []
    for chunk in str(value).replace(",", " ").split():
        item = chunk.strip()
        if item:
            modules.append(item)
    return modules or None


def resolve_lora_target_modules(
    target_modules: str | list[str] | tuple[str, ...] | None,
    *,
    include_lm_head: bool = False,
    include_embeddings: bool = False,
) -> list[str]:
    modules = list(
        parse_target_modules(target_modules) or DEFAULT_TRANSFORMER_TARGET_MODULES
    )
    if include_lm_head and "lm_head" not in modules:
        modules.append("lm_head")
    if include_embeddings and "embed_tokens" not in modules:
        modules.append("embed_tokens")
    return modules


def normalize_lora_train_scope(train_scope: str) -> str:
    scope = str(train_scope).strip()
    if scope not in LORA_TRAIN_SCOPE_CHOICES:
        raise ValueError(
            f"unknown LoRA train scope: {train_scope!r}; "
            f"expected one of {LORA_TRAIN_SCOPE_CHOICES}"
        )
    return scope


def lora_scope_includes_embeddings(
    train_scope: str, *, perturb_embeddings: bool = False
) -> bool:
    """Return whether the LoRA scope includes token embedding LoRA targets."""

    scope = normalize_lora_train_scope(train_scope)
    return scope == LORA_FULL_SCOPE or bool(perturb_embeddings)


def parse_update_bank_rank(value: str | int | None) -> int | None:
    """Parse a user-provided update bank rank.

    ``None`` and ``auto`` both mean the caller should estimate the capacity from
    the training horizon. Numeric values are treated as manual overrides.
    """

    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("update_bank_rank must be positive")
        return value
    text = str(value).strip().lower()
    if text in {"", "auto", "none"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("update_bank_rank must be positive")
    return parsed


def ceil_update_bank_rank_to_vllm_lora_rank(rank: int) -> int:
    """Round an update bank rank up to a vLLM-supported max_lora_rank."""

    parsed = int(rank)
    if parsed <= 0:
        raise ValueError("update_bank_rank must be positive")
    for allowed_rank in VLLM_MAX_LORA_RANKS:
        if parsed <= allowed_rank:
            return allowed_rank
    raise ValueError(
        "update_bank_rank exceeds the largest vLLM-supported max_lora_rank "
        f"({VLLM_MAX_LORA_RANKS[-1]}): {parsed}"
    )


def estimate_update_bank_rank(
    *,
    rank: int,
    steps: int,
    nu: int,
    warmup_steps: int = 0,
    rank_multiplier: int = 1,
    extra_blocks: int = 1,
) -> int:
    """Estimate LoRA bank capacity in rank units.

    The bank consumes one rank slice per V refresh. When the step horizon is
    finite, reserve exactly the needed number of slices plus a small guard block.
    Open-ended runs cannot be estimated and must pass a manual bank rank.
    """

    base_rank = int(rank) * int(rank_multiplier)
    if base_rank <= 0:
        raise ValueError("rank must be positive")
    nu_i = validate_nu(nu)
    if int(extra_blocks) < 0:
        raise ValueError("extra_blocks must be non-negative")
    total_steps = int(steps) + int(warmup_steps)
    if total_steps <= 0:
        raise ValueError(
            "update_bank_rank=auto requires a finite positive step horizon; "
            "set steps or pass an explicit update_bank_rank for open-ended runs"
        )
    blocks = 1 if is_infinite_nu(nu_i) else (total_steps + nu_i - 1) // nu_i
    guard_blocks = int(extra_blocks) if blocks > 1 else 0
    return max(base_rank, (blocks + guard_blocks) * base_rank)


def resolve_update_bank_rank(
    value: str | int | None,
    *,
    rank: int,
    steps: int,
    nu: int,
    warmup_steps: int = 0,
    rank_multiplier: int = 1,
    extra_blocks: int = 1,
) -> tuple[int, bool]:
    """Resolve update bank rank and report whether it was auto-estimated."""

    manual = parse_update_bank_rank(value)
    if manual is not None:
        return ceil_update_bank_rank_to_vllm_lora_rank(manual), False
    estimated = estimate_update_bank_rank(
        rank=rank,
        steps=steps,
        nu=nu,
        warmup_steps=warmup_steps,
        rank_multiplier=rank_multiplier,
        extra_blocks=extra_blocks,
    )
    return ceil_update_bank_rank_to_vllm_lora_rank(estimated), True
