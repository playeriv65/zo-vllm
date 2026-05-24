import argparse
import cProfile
import io
import json
import os
import pstats
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".cache", "hf")
os.environ.setdefault("HF_HOME", os.path.join(CACHE_ROOT, "home"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(CACHE_ROOT, "datasets"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(CACHE_ROOT, "hub"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(CACHE_ROOT, "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("WANDB_MODE", "offline")
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from zo_vllm.core.lozo_controller import LOZOConfig, LOZOController
from zo_vllm.core.temp_lora_runtime import TempLoRARuntime
from zo_vllm.core.vllm_scorer import (
    VLLMScorer,
    compute_nll_from_prompt_logprobs_detailed,
)


MODE_CHOICES = (
    "generate_only",
    "base",
    "direct_worker_base",
    "static_lora",
    "rewrite_lora_only",
    "score_with_rewrite",
    "direct_worker_score_with_rewrite",
)


def prepare_sst2_prompts(seed: int, num_samples: int) -> list[str]:
    rng = np.random.default_rng(seed)
    dataset = load_dataset("glue", "sst2", split="train")
    if num_samples < len(dataset):
        indices = rng.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)
    return [f"{item['sentence']} It was" for item in dataset]


def tokenize_prompts(tokenizer: Any, prompts: list[str]) -> list[list[int]]:
    return [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]


def extend_prompt_token_ids(
    prompt_token_ids: list[list[int]],
    min_prompt_tokens: int,
) -> list[list[int]]:
    if min_prompt_tokens <= 0:
        return prompt_token_ids
    extended = []
    for token_ids in prompt_token_ids:
        if len(token_ids) >= min_prompt_tokens:
            extended.append(token_ids)
            continue
        if not token_ids:
            raise ValueError("cannot extend an empty token id prompt")
        head = token_ids[:1]
        body = token_ids[1:] or token_ids
        repeated_body = []
        while len(head) + len(repeated_body) < min_prompt_tokens:
            repeated_body.extend(body)
        extended.append((head + repeated_body)[:min_prompt_tokens])
    return extended


def summarize_prompt_token_lengths(prompt_token_ids: list[list[int]]) -> dict[str, float | int]:
    lengths = [len(item) for item in prompt_token_ids]
    summary = summarize([float(item) for item in lengths])
    return {
        **summary,
        "sum": int(sum(lengths)),
        "count": int(len(lengths)),
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class GpuUtilMonitor:
    def __init__(self, gpu_id: str, interval_s: float) -> None:
        self.gpu_id = gpu_id
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> "GpuUtilMonitor":
        if self.interval_s <= 0 or not self.gpu_id:
            return self
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 2))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._query_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop_event.wait(self.interval_s)

    def _query_once(self) -> dict[str, float] | None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    self.gpu_id,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return None
        line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            return None
        try:
            return {
                "t_s": time.perf_counter() - self._t0,
                "utilization_gpu_pct": float(parts[0]),
                "memory_used_mib": float(parts[1]),
            }
        except ValueError:
            return None


def summarize_gpu_samples(samples: list[dict[str, float]]) -> dict[str, Any]:
    utils = [sample["utilization_gpu_pct"] for sample in samples]
    mem = [sample["memory_used_mib"] for sample in samples]
    return {
        "num_samples": len(samples),
        "utilization_gpu_pct": summarize(utils),
        "memory_used_mib": summarize(mem),
    }


def default_monitor_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible.split(",")[0].strip()
    return "0"


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(modes) - set(MODE_CHOICES))
    if unknown:
        raise SystemExit(f"unknown mode(s): {unknown}; valid modes: {','.join(MODE_CHOICES)}")
    if not modes:
        raise SystemExit("--modes must select at least one mode")
    return modes


def score_base_detailed(
    llm: LLM,
    tokenizer: Any,
    sampling_params: SamplingParams,
    prompts: list[str],
) -> tuple[float, dict[str, float | int]]:
    generate_t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generate_s = time.perf_counter() - generate_t0
    loss, stats = compute_nll_from_prompt_logprobs_detailed(outputs, tokenizer)
    return loss, {
        "score_generate_s": generate_s,
        "score_postprocess_s": stats["postprocess_s"],
        "score_num_outputs": stats["num_outputs"],
        "score_num_prompt_positions": stats["num_prompt_positions"],
        "score_num_loss_tokens": stats["num_loss_tokens"],
    }


def score_direct_worker_detailed(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    *,
    lora_ids: list[int] | None,
    max_logits_tokens: int,
    loss_impl: str,
) -> tuple[float, dict[str, float | int | str]]:
    score_t0 = time.perf_counter()
    result = llm.llm_engine.model_executor.collective_rpc(
        "zo_score_prompt_token_ids",
        kwargs={
            "prompt_token_ids": prompt_token_ids,
            "lora_ids": lora_ids,
            "max_logits_tokens": max_logits_tokens,
            "loss_impl": loss_impl,
        },
        single_value=True,
    )
    score_s = time.perf_counter() - score_t0
    return float(result["loss"]), {
        "score_direct_worker_s": score_s,
        "score_num_outputs": int(result["num_reqs"]),
        "score_num_prompt_positions": int(result["num_prompt_tokens"]),
        "score_num_loss_tokens": int(result["num_tokens"]),
        "score_num_tokens_padded": int(result["num_tokens_padded"]),
        "score_num_active_loras": int(result["num_active_loras"]),
        "score_cudagraph_mode": str(result["cudagraph_mode"]),
        "score_loss_impl": str(result.get("loss_impl", loss_impl)),
    }


def split_direct_worker_losses(
    result_loss_payload: dict[str, Any],
    split: int,
) -> tuple[float, float]:
    nll_sums = result_loss_payload["request_nll_sums"]
    num_tokens = result_loss_payload["request_num_tokens"]
    plus_nll = float(sum(nll_sums[:split]))
    plus_tokens = int(sum(num_tokens[:split]))
    minus_nll = float(sum(nll_sums[split:]))
    minus_tokens = int(sum(num_tokens[split:]))
    return plus_nll / plus_tokens, minus_nll / minus_tokens


def score_direct_worker_plus_minus_detailed(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    *,
    plus_id: int,
    minus_id: int,
    max_logits_tokens: int,
    loss_impl: str,
) -> tuple[float, float, dict[str, float | int | str]]:
    batch_token_ids = prompt_token_ids + prompt_token_ids
    lora_ids = [plus_id] * len(prompt_token_ids) + [minus_id] * len(prompt_token_ids)
    score_t0 = time.perf_counter()
    result = llm.llm_engine.model_executor.collective_rpc(
        "zo_score_prompt_token_ids",
        kwargs={
            "prompt_token_ids": batch_token_ids,
            "lora_ids": lora_ids,
            "max_logits_tokens": max_logits_tokens,
            "loss_impl": loss_impl,
        },
        single_value=True,
    )
    score_s = time.perf_counter() - score_t0
    loss_plus, loss_minus = split_direct_worker_losses(result, len(prompt_token_ids))
    return loss_plus, loss_minus, {
        "score_direct_worker_s": score_s,
        "score_num_outputs": int(result["num_reqs"]),
        "score_num_prompt_positions": int(result["num_prompt_tokens"]),
        "score_num_loss_tokens": int(result["num_tokens"]),
        "score_num_tokens_padded": int(result["num_tokens_padded"]),
        "score_num_active_loras": int(result["num_active_loras"]),
        "score_cudagraph_mode": str(result["cudagraph_mode"]),
        "score_loss_impl": str(result.get("loss_impl", loss_impl)),
    }


def sample_and_build_lora(
    controller: LOZOController,
    random_seed: int,
    output_device: str,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, float],
]:
    t0 = time.perf_counter()
    directions_2d, _directions_1d = controller.sample_direction(random_seed)
    direction_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    plus_A, plus_B = controller.build_temp_lora_tensors(
        directions_2d,
        sign=+1,
        output_device=output_device,
    )
    minus_A, minus_B = controller.build_temp_lora_tensors(
        directions_2d,
        sign=-1,
        output_device=output_device,
    )
    build_lora_s = time.perf_counter() - t0
    return directions_2d, plus_A, plus_B, minus_A, minus_B, {
        "direction_s": direction_s,
        "build_lora_s": build_lora_s,
    }


def update_temp_lora(
    temp_lora: TempLoRARuntime,
    plus_A: dict[str, torch.Tensor],
    plus_B: dict[str, torch.Tensor],
    minus_A: dict[str, torch.Tensor],
    minus_B: dict[str, torch.Tensor],
    step: int,
) -> float:
    synchronize()
    t0 = time.perf_counter()
    temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B, step=step)
    synchronize()
    return time.perf_counter() - t0


def make_controller_and_lora(
    *,
    args: argparse.Namespace,
    llm: LLM,
    model_name: str,
) -> tuple[LOZOController, TempLoRARuntime]:
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    config = LOZOConfig(
        rank=args.rank,
        eps=args.eps,
        lr=args.lr,
        step_interval=args.step_interval,
        master_device="cuda" if torch.cuda.is_available() else "cpu",
        random_device=args.zo_random_device,
        train_scope="lora_only",
    )
    controller = LOZOController(hf_model, config)
    temp_lora = TempLoRARuntime(
        rank=args.rank,
        num_layers=hf_model.config.num_hidden_layers,
        residency="gpu",
        injection="direct",
        llm=llm,
        base_model_name=model_name,
        hidden_size=hf_model.config.hidden_size,
        ffn_dim=hf_model.config.ffn_dim,
    )
    temp_lora.register_slots()
    return controller, temp_lora


def init_static_lora(
    *,
    controller: LOZOController,
    temp_lora: TempLoRARuntime,
    seed: int,
    output_device: str,
) -> dict[str, float]:
    _, plus_A, plus_B, minus_A, minus_B, build_timing = sample_and_build_lora(
        controller,
        random_seed=seed,
        output_device=output_device,
    )
    build_timing["lora_update_s"] = update_temp_lora(
        temp_lora,
        plus_A,
        plus_B,
        minus_A,
        minus_B,
        step=0,
    )
    return build_timing


def record_timing(timing: dict[str, list[float]], key: str, value: float) -> None:
    timing.setdefault(key, []).append(float(value))


def write_python_profile(
    *,
    profile: cProfile.Profile,
    output_dir: str,
    mode: str,
    sort_by: str,
    top_n: int,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, f"{mode}.pstats")
    txt_path = os.path.join(output_dir, f"{mode}_top.txt")
    profile.dump_stats(stats_path)
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(sort_by)
    stats.print_stats(top_n)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(stream.getvalue())
    return {"pstats": stats_path, "top_txt": txt_path}


def run_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: Any,
    prompts: list[str],
    prompt_token_ids: list[list[int]],
    sampling_params: SamplingParams,
    generate_only_sampling_params: SamplingParams,
    controller: LOZOController | None,
    temp_lora: TempLoRARuntime | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    scorer = VLLMScorer(
        llm,
        tokenizer,
        max_tokens=args.max_tokens,
        direct_prompt_nll=bool(int(args.direct_prompt_nll)),
    )
    timing: dict[str, list[float]] = {}
    losses: list[dict[str, float | int]] = []
    output_device = "cuda"
    profile = cProfile.Profile() if args.python_profile_dir else None
    scorer.sampling_params = sampling_params

    if mode in {
        "static_lora",
        "score_with_rewrite",
        "rewrite_lora_only",
        "direct_worker_score_with_rewrite",
    }:
        if controller is None or temp_lora is None:
            raise RuntimeError(f"{mode} requires controller and temp_lora")

    if mode == "static_lora":
        static_timing = init_static_lora(
            controller=controller,
            temp_lora=temp_lora,
            seed=args.seed,
            output_device=output_device,
        )
    else:
        static_timing = {}

    print(
        f"[microbench] mode={mode} steps={args.steps} warmup_steps={args.warmup_steps} "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    with GpuUtilMonitor(args.monitor_gpu, args.gpu_monitor_interval) as gpu_monitor:
        for step in range(args.steps + args.warmup_steps):
            is_warmup = step < args.warmup_steps
            logical_step = step - args.warmup_steps + 1
            random_seed = int(rng.integers(1_000_000_000))
            synchronize()
            if profile is not None and not is_warmup:
                profile.enable()
            step_t0 = time.perf_counter()
            try:
                if mode == "generate_only":
                    generate_t0 = time.perf_counter()
                    llm.generate(prompts + prompts, generate_only_sampling_params, use_tqdm=False)
                    generate_s = time.perf_counter() - generate_t0
                    synchronize()
                    step_s = time.perf_counter() - step_t0
                    if not is_warmup:
                        record_timing(timing, "generate_s", generate_s)
                        record_timing(timing, "step_s", step_s)

                elif mode == "base":
                    loss, detail = score_base_detailed(llm, tokenizer, sampling_params, prompts + prompts)
                    synchronize()
                    step_s = time.perf_counter() - step_t0
                    if not is_warmup:
                        record_timing(timing, "score_generate_s", float(detail["score_generate_s"]))
                        record_timing(timing, "score_postprocess_s", float(detail["score_postprocess_s"]))
                        record_timing(timing, "step_s", step_s)
                        losses.append({"step": logical_step, "loss": float(loss)})

                elif mode == "direct_worker_base":
                    loss, detail = score_direct_worker_detailed(
                        llm,
                        prompt_token_ids + prompt_token_ids,
                        lora_ids=None,
                        max_logits_tokens=args.direct_worker_max_logits_tokens,
                        loss_impl=args.direct_worker_loss_impl,
                    )
                    synchronize()
                    step_s = time.perf_counter() - step_t0
                    if not is_warmup:
                        record_timing(timing, "score_direct_worker_s", float(detail["score_direct_worker_s"]))
                        record_timing(timing, "step_s", step_s)
                        losses.append({"step": logical_step, "loss": float(loss)})

                elif mode == "static_lora":
                    loss_plus, loss_minus, detail = scorer.score_plus_minus_detailed(prompts, temp_lora)
                    synchronize()
                    step_s = time.perf_counter() - step_t0
                    if not is_warmup:
                        record_timing(timing, "score_request_build_s", detail["score_request_build_s"])
                        record_timing(timing, "score_generate_s", detail["score_generate_s"])
                        record_timing(timing, "score_postprocess_s", detail["score_postprocess_s"])
                        record_timing(timing, "step_s", step_s)
                        losses.append({
                            "step": logical_step,
                            "loss_plus": float(loss_plus),
                            "loss_minus": float(loss_minus),
                        })

                elif mode == "direct_worker_score_with_rewrite":
                    _, plus_A, plus_B, minus_A, minus_B, build_timing = sample_and_build_lora(
                        controller,
                        random_seed=random_seed,
                        output_device=output_device,
                    )
                    lora_update_s = update_temp_lora(
                        temp_lora,
                        plus_A,
                        plus_B,
                        minus_A,
                        minus_B,
                        step=step + 1,
                    )
                    score_t0 = time.perf_counter()
                    loss_plus, loss_minus, detail = score_direct_worker_plus_minus_detailed(
                        llm,
                        prompt_token_ids,
                        plus_id=temp_lora.plus_id,
                        minus_id=temp_lora.minus_id,
                        max_logits_tokens=args.direct_worker_max_logits_tokens,
                        loss_impl=args.direct_worker_loss_impl,
                    )
                    synchronize()
                    score_s = time.perf_counter() - score_t0
                    step_s = time.perf_counter() - step_t0
                    if not is_warmup:
                        record_timing(timing, "direction_s", build_timing["direction_s"])
                        record_timing(timing, "build_lora_s", build_timing["build_lora_s"])
                        record_timing(timing, "lora_update_s", lora_update_s)
                        record_timing(timing, "score_s", score_s)
                        record_timing(timing, "score_direct_worker_s", float(detail["score_direct_worker_s"]))
                        record_timing(timing, "step_s", step_s)
                        losses.append({
                            "step": logical_step,
                            "loss_plus": float(loss_plus),
                            "loss_minus": float(loss_minus),
                        })

                elif mode == "rewrite_lora_only":
                    _, plus_A, plus_B, minus_A, minus_B, build_timing = sample_and_build_lora(
                        controller,
                        random_seed=random_seed,
                        output_device=output_device,
                    )
                    lora_update_s = update_temp_lora(
                        temp_lora,
                        plus_A,
                        plus_B,
                        minus_A,
                        minus_B,
                        step=step + 1,
                    )
                    step_s = time.perf_counter() - step_t0
                    if not is_warmup:
                        record_timing(timing, "direction_s", build_timing["direction_s"])
                        record_timing(timing, "build_lora_s", build_timing["build_lora_s"])
                        record_timing(timing, "lora_update_s", lora_update_s)
                        record_timing(timing, "step_s", step_s)

                elif mode == "score_with_rewrite":
                    _, plus_A, plus_B, minus_A, minus_B, build_timing = sample_and_build_lora(
                        controller,
                        random_seed=random_seed,
                        output_device=output_device,
                    )
                    lora_update_s = update_temp_lora(
                        temp_lora,
                        plus_A,
                        plus_B,
                        minus_A,
                        minus_B,
                        step=step + 1,
                    )
                    score_t0 = time.perf_counter()
                    loss_plus, loss_minus, detail = scorer.score_plus_minus_detailed(prompts, temp_lora)
                    synchronize()
                    score_s = time.perf_counter() - score_t0
                    step_s = time.perf_counter() - step_t0
                    if not is_warmup:
                        record_timing(timing, "direction_s", build_timing["direction_s"])
                        record_timing(timing, "build_lora_s", build_timing["build_lora_s"])
                        record_timing(timing, "lora_update_s", lora_update_s)
                        record_timing(timing, "score_s", score_s)
                        record_timing(timing, "score_request_build_s", detail["score_request_build_s"])
                        record_timing(timing, "score_generate_s", detail["score_generate_s"])
                        record_timing(timing, "score_postprocess_s", detail["score_postprocess_s"])
                        record_timing(timing, "step_s", step_s)
                        losses.append({
                            "step": logical_step,
                            "loss_plus": float(loss_plus),
                            "loss_minus": float(loss_minus),
                        })
            finally:
                if profile is not None and not is_warmup:
                    profile.disable()

            if (
                not is_warmup
                and args.progress_interval > 0
                and logical_step % args.progress_interval == 0
            ):
                latest_step_s = timing.get("step_s", [0.0])[-1]
                print(
                    f"[microbench] mode={mode} step={logical_step}/{args.steps} "
                    f"step_s={latest_step_s:.6f}",
                    flush=True,
                )

    summary = {key: summarize(values) for key, values in sorted(timing.items())}
    step_mean = summary.get("step_s", {}).get("mean", 0.0)
    effective_requests = 2 * args.batch_size
    effective_request_s = effective_requests / step_mean if step_mean > 0 else 0.0
    profile_outputs = {}
    if profile is not None:
        profile_outputs = write_python_profile(
            profile=profile,
            output_dir=args.python_profile_dir,
            mode=mode,
            sort_by=args.python_profile_sort,
            top_n=args.python_profile_top,
        )
    return {
        "mode": mode,
        "static_lora_setup_timing": static_timing,
        "effective_requests_per_step": effective_requests,
        "effective_requests_per_s": effective_request_s,
        "timing": summary,
        "python_profile": profile_outputs,
        "gpu_monitor": summarize_gpu_samples(gpu_monitor.samples),
        "gpu_monitor_samples": gpu_monitor.samples[: args.max_gpu_monitor_records],
        "loss_samples": losses[: min(len(losses), args.max_loss_records)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 q=1 vLLM scoring microbenchmark."
    )
    parser.add_argument("--model", default="facebook/opt-2.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--prompt-input", choices=["text", "token_ids"], default="text")
    parser.add_argument("--min-prompt-tokens", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--detokenize", choices=["0", "1"], default="0")
    parser.add_argument("--flat-logprobs", choices=["0", "1"], default="0")
    parser.add_argument("--direct-prompt-nll", choices=["0", "1"], default="1")
    parser.add_argument("--direct-worker-max-logits-tokens", type=int, default=8192)
    parser.add_argument("--direct-worker-loss-impl", choices=["logprobs", "cross_entropy"], default="logprobs")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--step-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zo-random-device", choices=["cuda"], default="cuda")
    parser.add_argument("--batch-invariant", choices=["0", "1"], default="0")
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--gpu-monitor-interval", type=float, default=0.0)
    parser.add_argument("--monitor-gpu", default=default_monitor_gpu())
    parser.add_argument(
        "--modes",
        default="base,static_lora,rewrite_lora_only,score_with_rewrite",
        help=f"Comma-separated subset of: {','.join(MODE_CHOICES)}",
    )
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--max-loss-records", type=int, default=10)
    parser.add_argument("--max-gpu-monitor-records", type=int, default=2000)
    parser.add_argument("--python-profile-dir", default=None)
    parser.add_argument("--python-profile-sort", default="cumtime")
    parser.add_argument("--python-profile-top", type=int, default=80)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Phase 3 vLLM scoring microbenchmarks")
    if args.num_samples < args.batch_size:
        raise SystemExit("--num-samples must be greater than or equal to --batch-size")

    os.environ["VLLM_BATCH_INVARIANT"] = args.batch_invariant
    modes = parse_modes(args.modes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT,
        "phase3",
        "results",
        timestamp,
        "microbench_vllm_scoring",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"[microbench] model={args.model} modes={','.join(modes)} "
        f"batch_size={args.batch_size} steps={args.steps} warmup_steps={args.warmup_steps} "
        f"prompt_input={args.prompt_input} max_tokens={args.max_tokens} "
        f"detokenize={args.detokenize} flat_logprobs={args.flat_logprobs} "
        f"direct_prompt_nll={args.direct_prompt_nll} "
        f"rank={args.rank} eps={args.eps} enforce_eager={args.enforce_eager} "
        f"batch_invariant={args.batch_invariant} gpu_memory_utilization={args.gpu_memory_utilization}",
        flush=True,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text_prompts = prepare_sst2_prompts(args.seed, args.num_samples)[: args.batch_size]
    prompt_token_ids = tokenize_prompts(tokenizer, text_prompts)
    if args.min_prompt_tokens > 0 and args.prompt_input != "token_ids":
        raise SystemExit("--min-prompt-tokens requires --prompt-input token_ids")
    prompt_token_ids = extend_prompt_token_ids(prompt_token_ids, args.min_prompt_tokens)
    prompt_length_summary = summarize_prompt_token_lengths(prompt_token_ids)
    prompts = prompt_token_ids if args.prompt_input == "token_ids" else text_prompts
    llm = LLM(
        model=args.model,
        enforce_eager=bool(int(args.enforce_eager)),
        enable_lora=True,
        max_lora_rank=args.rank,
        max_loras=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        prompt_logprobs=0 if bool(int(args.direct_prompt_nll)) else 1,
        detokenize=bool(int(args.detokenize)),
        flat_logprobs=bool(int(args.flat_logprobs)),
        skip_clone=True,
        extra_args=(
            {"zo_direct_prompt_nll": True}
            if bool(int(args.direct_prompt_nll))
            else None
        ),
    )
    generate_only_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max(1, args.max_tokens),
        prompt_logprobs=None,
        detokenize=bool(int(args.detokenize)),
        skip_clone=True,
    )

    needs_lora = any(
        mode
        in {
            "static_lora",
            "score_with_rewrite",
            "rewrite_lora_only",
            "direct_worker_score_with_rewrite",
        }
        for mode in modes
    )
    controller = None
    temp_lora = None
    if needs_lora:
        controller, temp_lora = make_controller_and_lora(
            args=args,
            llm=llm,
            model_name=args.model,
        )

    results = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "batch_size": args.batch_size,
            "prompt_input": args.prompt_input,
            "min_prompt_tokens": args.min_prompt_tokens,
            "prompt_token_lengths": prompt_length_summary,
            "effective_prompt_tokens_per_step": int(2 * prompt_length_summary["sum"]),
            "max_tokens": args.max_tokens,
            "detokenize": args.detokenize,
            "flat_logprobs": args.flat_logprobs,
            "direct_prompt_nll": args.direct_prompt_nll,
            "direct_worker_max_logits_tokens": args.direct_worker_max_logits_tokens,
            "direct_worker_loss_impl": args.direct_worker_loss_impl,
            "effective_requests_per_step": 2 * args.batch_size,
            "num_samples": args.num_samples,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "rank": args.rank,
            "eps": args.eps,
            "lr": args.lr,
            "step_interval": args.step_interval,
            "seed": args.seed,
            "zo_random_device": args.zo_random_device,
            "batch_invariant": args.batch_invariant,
            "enforce_eager": args.enforce_eager,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_monitor_interval": args.gpu_monitor_interval,
            "monitor_gpu": args.monitor_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "wandb_mode": os.environ.get("WANDB_MODE", ""),
        },
        "modes": {},
    }

    try:
        for mode in modes:
            results["modes"][mode] = run_mode(
                mode=mode,
                args=args,
                llm=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                generate_only_sampling_params=generate_only_sampling_params,
                controller=controller,
                temp_lora=temp_lora,
            )
    finally:
        if temp_lora is not None:
            temp_lora.cleanup()

    output_path = os.path.join(output_dir, "microbench_vllm_scoring.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[microbench] wrote {output_path}", flush=True)
    for mode, mode_result in results["modes"].items():
        step_mean = mode_result["timing"].get("step_s", {}).get("mean", 0.0)
        request_s = mode_result["effective_requests_per_s"]
        print(
            f"[microbench] summary mode={mode} step_s_mean={step_mean:.6f} "
            f"effective_requests_per_s={request_s:.2f}",
            flush=True,
        )
        gpu_summary = mode_result.get("gpu_monitor", {}).get("utilization_gpu_pct", {})
        if gpu_summary:
            print(
                f"[microbench] gpu mode={mode} util_mean={gpu_summary.get('mean', 0.0):.1f}% "
                f"util_max={gpu_summary.get('max', 0.0):.1f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
