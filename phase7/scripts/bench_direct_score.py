from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from zo_vllm.experiment.infra.env import (  # noqa: E402
    configure_hf_cache,
    configure_vllm_training_env,
)

configure_hf_cache(PROJECT_ROOT)
configure_vllm_training_env()

from transformers import AutoTokenizer  # noqa: E402

from zo_vllm.core.direct_worker_scorer import score_token_id_groups  # noqa: E402
from zo_vllm.experiment.infra.stats import summarize, summarize_tail  # noqa: E402
from zo_vllm.training.task_batches import (  # noqa: E402
    build_zo_task_batch,
    load_objective_rows,
)
from zo_vllm.tasks.tokenization import configure_opt_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="facebook/opt-13b")
    parser.add_argument("--prequant-model", default=None)
    parser.add_argument("--vllm-quantization", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeat-sides", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--enable-lora", choices=["0", "1"], default="0")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-loras", type=int, default=2)
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-logits-tokens", type=int, default=8192)
    parser.add_argument("--loss-impl", choices=["logprobs", "cross_entropy"], default="logprobs")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.repeat_sides <= 0:
        raise SystemExit("--repeat-sides must be positive")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    model_name = args.prequant_model or args.model_name
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    configure_opt_tokenizer(tokenizer, model_name)
    train_rows, _dev_rows, _valid_rows = load_objective_rows(
        "sst2_classification",
        data_seed=args.data_seed,
        num_train=max(args.num_samples, args.batch_size),
        num_dev=1,
        num_eval=1,
    )
    batch_rows = train_rows[: args.batch_size]
    zo_batch = build_zo_task_batch(
        batch_rows,
        tokenizer,
        objective_name="sst2_classification",
        max_length=args.max_model_len,
    )
    token_groups = list(zo_batch.token_id_groups) * int(args.repeat_sides)
    labels = (
        None
        if zo_batch.labels is None
        else list(zo_batch.labels) * int(args.repeat_sides)
    )
    needed_reqs = len(token_groups)
    max_num_seqs = args.max_num_seqs or needed_reqs
    if max_num_seqs < needed_reqs:
        raise SystemExit(
            f"--max-num-seqs={max_num_seqs} is too small for {needed_reqs} requests"
        )

    from vllm import LLM

    llm_kwargs = {
        "model": model_name,
        "enforce_eager": bool(int(args.enforce_eager)),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "max_num_seqs": int(max_num_seqs),
        "enable_lora": bool(int(args.enable_lora)),
    }
    if bool(int(args.enable_lora)):
        llm_kwargs["max_lora_rank"] = int(args.max_lora_rank)
        llm_kwargs["max_loras"] = int(args.max_loras)
    if args.vllm_quantization:
        llm_kwargs["quantization"] = args.vllm_quantization

    print(
        "[phase7-direct-score] "
        f"model={model_name} quantization={args.vllm_quantization} "
        f"enable_lora={args.enable_lora} enforce_eager={args.enforce_eager} "
        f"batch_size={args.batch_size} repeat_sides={args.repeat_sides} "
        f"num_reqs={needed_reqs} max_num_seqs={max_num_seqs} "
        f"steps={args.steps} warmup_steps={args.warmup_steps}",
        flush=True,
    )
    init_t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    init_s = time.perf_counter() - init_t0

    timings: list[float] = []
    last_result = None
    total_steps = int(args.warmup_steps) + int(args.steps)
    measured_t0 = None
    for step in range(1, total_steps + 1):
        t0 = time.perf_counter()
        result = score_token_id_groups(
            llm,
            token_groups,
            lora_ids=None,
            labels=labels,
            max_logits_tokens=int(args.max_logits_tokens),
            loss_impl=args.loss_impl,
        )
        elapsed = time.perf_counter() - t0
        if step > args.warmup_steps:
            if measured_t0 is None:
                measured_t0 = t0
            timings.append(elapsed)
        last_result = result
        if step > args.warmup_steps and len(timings) % 20 == 0:
            print(
                f"[phase7-direct-score] step={len(timings)} "
                f"score_s={elapsed:.6f}",
                flush=True,
            )
    measured_total_s = 0.0 if measured_t0 is None else time.perf_counter() - measured_t0
    summary = {
        "config": {
            **vars(args),
            "resolved_model": model_name,
            "num_reqs": needed_reqs,
            "max_num_seqs_resolved": max_num_seqs,
        },
        "init_s": init_s,
        "total_s": measured_total_s,
        "score_s": summarize(timings),
        "tail_100_score_s": summarize_tail(timings, 100),
        "last_result_summary": {
            "num_reqs": None if last_result is None else int(last_result["num_reqs"]),
            "num_prompt_tokens": None
            if last_result is None
            else int(last_result["num_prompt_tokens"]),
            "num_tokens": None if last_result is None else int(last_result["num_tokens"]),
            "num_active_loras": None
            if last_result is None
            else int(last_result["num_active_loras"]),
            "cudagraph_mode": None
            if last_result is None
            else str(last_result["cudagraph_mode"]),
            "profile_s": None if last_result is None else last_result.get("profile_s", {}),
            "profile_cuda_ms": None
            if last_result is None
            else last_result.get("profile_cuda_ms", {}),
        },
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"direct_score_bench_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        "[phase7-direct-score-result] "
        f"json={path} init_s={init_s:.6f} "
        f"mean_score_s={summary['score_s']['mean']:.6f} "
        f"tail100_score_s={summary['tail_100_score_s']['mean']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
