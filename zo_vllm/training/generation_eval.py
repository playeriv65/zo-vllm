"""Generation-based evaluation helpers for task metrics."""

from __future__ import annotations

from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from zo_vllm.core.lora_runtime import DIRECT_SLOT_PATH_PREFIX
from zo_vllm.tasks.squad import (
    encode_squad_generation_prompts,
    squad_exact_match,
    squad_f1,
)


def eval_squad_f1(
    llm,
    tokenizer,
    rows,
    *,
    max_length: int,
    max_new_tokens: int,
    lora_id: int | None = None,
    max_prompts_per_call: int = 16,
) -> dict[str, float] | None:
    if not rows:
        return None
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=["\n"],
    )
    total_f1 = 0.0
    total_em = 0.0
    total = 0
    for start in range(0, len(rows), max_prompts_per_call):
        sub_rows = rows[start : start + max_prompts_per_call]
        prompt_ids = encode_squad_generation_prompts(
            sub_rows,
            tokenizer,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
        lora_request = None
        if lora_id is not None:
            lora_request = [
                LoRARequest(
                    "lozo_clean",
                    int(lora_id),
                    f"{DIRECT_SLOT_PATH_PREFIX}/{int(lora_id)}",
                )
                for _ in prompts
            ]
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )
        for row, output in zip(sub_rows, outputs):
            prediction = output.outputs[0].text.strip()
            total_f1 += squad_f1(prediction, row.answers)
            total_em += squad_exact_match(prediction, row.answers)
            total += 1
    return {
        "f1": float(total_f1 / max(total, 1)),
        "em": float(total_em / max(total, 1)),
    }
