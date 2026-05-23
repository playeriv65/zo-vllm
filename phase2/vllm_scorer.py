"""
vLLM Scorer - Compute loss using vLLM with prompt logprobs.

Reuses Phase 1 loss computation logic.
"""

from typing import List, Dict, Optional
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            prompt_logprobs=1,
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
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            lora_request=LoRARequest(lora_name, lora_id, lora_path),
            use_tqdm=False,
        )
        return compute_nll_from_prompt_logprobs(outputs, self.tokenizer)
    
    def score_base(self, prompts: List[str]) -> float:
        """Compute NLL without LoRA (base model only)."""
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            use_tqdm=False,
        )
        return compute_nll_from_prompt_logprobs(outputs, self.tokenizer)
    
    def score_plus_minus(
        self,
        prompts: List[str],
        temp_lora_runtime,
    ) -> tuple[float, float]:
        """
        Compute NLL with plus/minus LoRA for LOZO.
        
        Args:
            prompts: List of prompts
            temp_lora_runtime: TempLoRARuntime instance
        
        Returns:
            (loss_plus, loss_minus)
        """
        plus_name, plus_id, plus_path = temp_lora_runtime.get_plus_request_info()
        minus_name, minus_id, minus_path = temp_lora_runtime.get_minus_request_info()
        load_inplace = getattr(temp_lora_runtime, "request_load_inplace", True)
        
        outputs = self.llm.generate(
            list(prompts) + list(prompts),
            self.sampling_params,
            lora_request=[
                LoRARequest(plus_name, plus_id, plus_path, load_inplace=load_inplace)
                for _ in prompts
            ] + [
                LoRARequest(minus_name, minus_id, minus_path, load_inplace=load_inplace)
                for _ in prompts
            ],
            use_tqdm=False,
        )
        split = len(prompts)
        loss_plus = compute_nll_from_prompt_logprobs(outputs[:split], self.tokenizer)
        loss_minus = compute_nll_from_prompt_logprobs(outputs[split:], self.tokenizer)
        
        return loss_plus, loss_minus
