from transformers import AutoTokenizer

from zo_vllm.tasks.tokenization import (
    OPT_BOS_LOZO,
    OPT_BOS_NATIVE,
    configure_opt_tokenizer,
    tokenizer_from_pretrained_kwargs,
)


def _load_opt_tokenizer(mode: str):
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        **tokenizer_from_pretrained_kwargs(model_name, opt_bos_mode=mode),
    )
    configure_opt_tokenizer(tokenizer, model_name, opt_bos_mode=mode)
    return tokenizer


def test_opt_native_bos_keeps_hf_default():
    tokenizer = _load_opt_tokenizer(OPT_BOS_NATIVE)

    assert tokenizer.encode("hello")[0] == 2


def test_opt_lozo_bos_matches_mezo_reproduction_setting():
    tokenizer = _load_opt_tokenizer(OPT_BOS_LOZO)

    assert tokenizer.encode("hello")[0] == 0
