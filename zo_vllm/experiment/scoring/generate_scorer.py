"""Legacy generate(prompt_logprobs) scorer used by validation and microbenchmarks."""

import time
from typing import List
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ZO_PROMPT_NLL_KEY = "__zo_prompt_nll__"


def is_direct_prompt_nll(prompt_logprobs) -> bool:
    return isinstance(prompt_logprobs, dict) and bool(prompt_logprobs.get(ZO_PROMPT_NLL_KEY))


def compute_nll_from_prompt_logprobs(outputs, tokenizer) -> float:
    """
    Compute negative log-likelihood from vLLM outputs with prompt_logprobs.

    Returns AVERAGE of NLL (not sum), to match HF baseline behavior (reduction="mean").

    Args:
        outputs: vLLM generate outputs
        tokenizer: Tokenizer for decoding

    Returns:
        avg_nll: Average NLL over all predicted tokens in the batch
    """
    total_nll = 0.0
    total_tokens = 0

    for output in outputs:
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            continue

        if is_direct_prompt_nll(prompt_logprobs):
            total_nll += float(prompt_logprobs["nll_sum"])
            total_tokens += int(prompt_logprobs["num_tokens"])
            continue

        # We start from index 1 because the first token does not have a prefix to predict it.
        # This matches how Causal LM loss is computed (predicting token j given 0..j-1)
        for j in range(1, len(prompt_logprobs)):
            token_logprobs = prompt_logprobs[j]
            if token_logprobs is None:
                continue

            if not hasattr(output, "prompt_token_ids") or output.prompt_token_ids is None:
                continue

            token_id = output.prompt_token_ids[j]
            if token_id in token_logprobs:
                val = token_logprobs[token_id]
                logprob = val.logprob if hasattr(val, "logprob") else float(val)
                total_nll += -logprob
                total_tokens += 1

    if total_tokens == 0:
        return 0.0
    return total_nll / total_tokens


def compute_nll_from_prompt_logprobs_detailed(outputs, tokenizer) -> tuple[float, dict]:
    """Compute NLL and return lightweight postprocess counters/timing."""
    t0 = time.perf_counter()
    total_nll = 0.0
    total_tokens = 0
    total_positions = 0

    for output in outputs:
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            continue

        if is_direct_prompt_nll(prompt_logprobs):
            num_tokens = int(prompt_logprobs["num_tokens"])
            total_nll += float(prompt_logprobs["nll_sum"])
            total_tokens += num_tokens
            total_positions += num_tokens
            continue

        prompt_token_ids = getattr(output, "prompt_token_ids", None)
        if prompt_token_ids is None:
            continue

        for j in range(1, len(prompt_logprobs)):
            total_positions += 1
            token_logprobs = prompt_logprobs[j]
            if token_logprobs is None:
                continue

            token_id = prompt_token_ids[j]
            if token_id in token_logprobs:
                val = token_logprobs[token_id]
                logprob = val.logprob if hasattr(val, "logprob") else float(val)
                total_nll += -logprob
                total_tokens += 1

    loss = total_nll / total_tokens if total_tokens else 0.0
    return loss, {
        "postprocess_s": time.perf_counter() - t0,
        "num_outputs": len(outputs),
        "num_prompt_positions": total_positions,
        "num_loss_tokens": total_tokens,
    }


class VLLMScorer:
    """
    Compute loss using vLLM engine.

    Supports:
    - Base model scoring
    - LoRA scoring (in-memory or file-based)
    - Plus/minus scoring for LOZO
    """

    def __init__(
        self,
        llm: LLM,
        tokenizer,
        max_tokens: int = 1,
        direct_prompt_nll: bool = True,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.direct_prompt_nll = direct_prompt_nll
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            prompt_logprobs=0 if direct_prompt_nll else 1,
            detokenize=False,
            skip_clone=True,
            extra_args=(
                {"zo_direct_prompt_nll": True} if direct_prompt_nll else None
            ),
        )

    def _encode_prompt(self, prompt: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(prompt, add_special_tokens=True))
        encoded = self.tokenizer(prompt, add_special_tokens=True)
        return list(encoded["input_ids"])

    def _direct_prompt_inputs(
        self, prompts: List[str]
    ) -> tuple[list[dict], list[SamplingParams]]:
        prompt_inputs = []
        sampling_params = []
        for prompt in prompts:
            token_ids = self._encode_prompt(prompt)
            labels = [-100] + token_ids[1:]
            prompt_inputs.append({"prompt_token_ids": token_ids})
            sampling_params.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    prompt_logprobs=0,
                    detokenize=False,
                    skip_clone=True,
                    extra_args={
                        "zo_direct_prompt_nll": True,
                        "zo_loss_labels": labels,
                    },
                )
            )
        return prompt_inputs, sampling_params

    def _generate(
        self,
        prompts: List[str],
        *,
        lora_request=None,
    ):
        if not self.direct_prompt_nll:
            return self.llm.generate(
                prompts,
                self.sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
        prompt_inputs, sampling_params = self._direct_prompt_inputs(prompts)
        return self.llm.generate(
            prompt_inputs,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

    def score_with_lora(
        self,
        prompts: List[str],
        lora_name: str,
        lora_id: int,
        lora_path: str,
    ) -> float:
        """
        Compute NLL with LoRA adapter.

        Args:
            prompts: List of prompts
            lora_name: LoRA name
            lora_id: LoRA integer ID
            lora_path: Path to LoRA (can be memory path)

        Returns:
            avg_nll: Average negative log-likelihood
        """
        outputs = self._generate(
            prompts,
            lora_request=LoRARequest(lora_name, lora_id, lora_path),
        )
        return compute_nll_from_prompt_logprobs(outputs, self.tokenizer)

    def score_base(self, prompts: List[str]) -> float:
        """Compute NLL without LoRA (base model only)."""
        outputs = self._generate(
            prompts,
        )
        return compute_nll_from_prompt_logprobs(outputs, self.tokenizer)

    def score_plus_minus(
        self,
        prompts: List[str],
        lora_update_runtime,
    ) -> tuple[float, float]:
        """
        Compute NLL with plus/minus LoRA for LOZO.

        Args:
            prompts: List of prompts
            lora_update_runtime: LoRAUpdateRuntime instance

        Returns:
            (loss_plus, loss_minus)
        """
        plus_name, plus_id, plus_path = lora_update_runtime.get_plus_request_info()
        minus_name, minus_id, minus_path = lora_update_runtime.get_minus_request_info()
        load_inplace = getattr(lora_update_runtime, "request_load_inplace", True)

        outputs = self._generate(
            list(prompts) + list(prompts),
            lora_request=[
                LoRARequest(plus_name, plus_id, plus_path, load_inplace=load_inplace)
                for _ in prompts
            ] + [
                LoRARequest(minus_name, minus_id, minus_path, load_inplace=load_inplace)
                for _ in prompts
            ],
        )
        split = len(prompts)
        loss_plus = compute_nll_from_prompt_logprobs(outputs[:split], self.tokenizer)
        loss_minus = compute_nll_from_prompt_logprobs(outputs[split:], self.tokenizer)

        return loss_plus, loss_minus

    def score_plus_minus_detailed(
        self,
        prompts: List[str],
        lora_update_runtime,
    ) -> tuple[float, float, dict]:
        """Compute plus/minus NLL with a timing breakdown for profiling."""
        plus_name, plus_id, plus_path = lora_update_runtime.get_plus_request_info()
        minus_name, minus_id, minus_path = lora_update_runtime.get_minus_request_info()
        load_inplace = getattr(lora_update_runtime, "request_load_inplace", True)
        prompt_list = list(prompts)

        request_build_t0 = time.perf_counter()
        request_prompts = prompt_list + prompt_list
        lora_requests = [
            LoRARequest(plus_name, plus_id, plus_path, load_inplace=load_inplace)
            for _ in prompt_list
        ] + [
            LoRARequest(minus_name, minus_id, minus_path, load_inplace=load_inplace)
            for _ in prompt_list
        ]
        request_build_s = time.perf_counter() - request_build_t0

        generate_t0 = time.perf_counter()
        outputs = self._generate(
            request_prompts,
            lora_request=lora_requests,
        )
        generate_s = time.perf_counter() - generate_t0

        split = len(prompt_list)
        loss_plus, plus_stats = compute_nll_from_prompt_logprobs_detailed(
            outputs[:split], self.tokenizer
        )
        loss_minus, minus_stats = compute_nll_from_prompt_logprobs_detailed(
            outputs[split:], self.tokenizer
        )
        postprocess_s = plus_stats["postprocess_s"] + minus_stats["postprocess_s"]
        return loss_plus, loss_minus, {
            "score_request_build_s": request_build_s,
            "score_generate_s": generate_s,
            "score_postprocess_s": postprocess_s,
            "score_postprocess_plus_s": plus_stats["postprocess_s"],
            "score_postprocess_minus_s": minus_stats["postprocess_s"],
            "score_num_outputs": plus_stats["num_outputs"] + minus_stats["num_outputs"],
            "score_num_prompt_positions": (
                plus_stats["num_prompt_positions"] + minus_stats["num_prompt_positions"]
            ),
            "score_num_loss_tokens": plus_stats["num_loss_tokens"] + minus_stats["num_loss_tokens"],
        }
