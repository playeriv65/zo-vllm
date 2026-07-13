from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase


OPT_BOS_NATIVE = "native"
OPT_BOS_LOZO = "lozo"
OPT_BOS_MODES = (OPT_BOS_NATIVE, OPT_BOS_LOZO)


def is_opt_model(model_name: str) -> bool:
    model_s = str(model_name).replace("\\", "/").lower()
    return any(
        part.startswith("opt-") or part.startswith("models--facebook--opt-")
        for part in model_s.split("/")
    )


def tokenizer_from_pretrained_kwargs(
    model_name: str,
    *,
    opt_bos_mode: str = OPT_BOS_NATIVE,
) -> dict[str, Any]:
    """Return tokenizer kwargs needed before from_pretrained construction."""

    mode = normalize_opt_bos_mode(opt_bos_mode)
    if is_opt_model(model_name) and mode == OPT_BOS_LOZO:
        return {"bos_token": "<s>"}
    return {}


def normalize_opt_bos_mode(mode: str) -> str:
    mode_s = str(mode).strip().lower()
    if mode_s not in OPT_BOS_MODES:
        raise ValueError(f"opt_bos_mode must be one of {OPT_BOS_MODES}, got {mode!r}")
    return mode_s


def configure_opt_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    *,
    opt_bos_mode: str = OPT_BOS_NATIVE,
) -> None:
    """Apply post-load tokenizer settings.

    OPT's HuggingFace-native tokenizer uses ``</s>`` (id 2) as BOS. LOZO/MeZO
    changed OPT to use ``<s>`` (id 0) for strict baseline reproduction. With
    newer Transformers versions, that LOZO setting must be passed during
    ``from_pretrained`` via ``tokenizer_from_pretrained_kwargs``; assigning
    ``bos_token_id`` here is not sufficient to change ``encode``.
    """

    mode = normalize_opt_bos_mode(opt_bos_mode)
    if is_opt_model(model_name) and mode == OPT_BOS_LOZO:
        tokenizer.bos_token = "<s>"


def single_token_id(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"verbalizer must be single token, got {text!r} -> {token_ids}"
        )
    return int(token_ids[0])
