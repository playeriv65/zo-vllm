"""Compare direct worker scoring with scheduler-mediated scoring."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from zo_vllm.experiment.infra.env import (  # noqa: E402
    configure_hf_cache,
    configure_vllm_training_env,
)

configure_hf_cache(str(PROJECT_ROOT))
configure_vllm_training_env()
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

from transformers import AutoTokenizer  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.sampling_params import RequestOutputKind, SamplingParams  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

from zo_vllm.training import build_objective_batch  # noqa: E402
from zo_vllm.training.task_batches import load_objective_rows  # noqa: E402
from zo_vllm.tasks.tokenization import configure_opt_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeat-sides", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-pool-size", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-logits-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument(
        "--scheduler-admission",
        choices=["standard", "paused_batch"],
        default="standard",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


async def score_direct(
    engine: AsyncLLM,
    token_groups: list[list[int]],
    labels: list[list[int]],
    *,
    max_logits_tokens: int,
) -> tuple[float, int]:
    results = await engine.collective_rpc(
        "zo_score_prompt_token_ids",
        kwargs={
            "prompt_token_ids": token_groups,
            "lora_ids": None,
            "labels": labels,
            "max_logits_tokens": int(max_logits_tokens),
            "loss_impl": "logprobs",
            "return_logits": False,
            "compute_token_nll": True,
            "return_request_nll_tensor": False,
            "return_device_tensors": False,
        },
    )
    if not results or not isinstance(results[0], dict):
        raise RuntimeError("direct collective RPC returned no worker score")
    result = results[0]
    nll_sum = float(result["nll_sum"])
    num_tokens = int(result["num_tokens"])
    return nll_sum, num_tokens


async def score_scheduled(
    engine: AsyncLLM,
    token_groups: list[list[int]],
    labels: list[list[int]],
    *,
    admission: str,
) -> tuple[float, int]:
    def build_params(token_labels: list[int]) -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=0,
            prompt_logprobs=0,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            skip_clone=True,
            extra_args={
                "zo_direct_prompt_nll": True,
                "zo_loss_labels": token_labels,
            },
        )

    def compact_result(final_output) -> tuple[float, int]:
        compact = final_output.prompt_logprobs
        if not isinstance(compact, dict) or not compact.get("__zo_prompt_nll__"):
            raise RuntimeError("scheduled request did not return compact prompt NLL")
        return float(compact["nll_sum"]), int(compact["num_tokens"])

    async def score_one(token_ids: list[int], token_labels: list[int]) -> tuple[float, int]:
        params = build_params(token_labels)
        final_output = None
        async for output in engine.generate(
            {"prompt_token_ids": token_ids},
            params,
            f"scheduler-overhead-{uuid.uuid4().hex}",
            priority=1000,
        ):
            final_output = output
        if final_output is None:
            raise RuntimeError("scheduled request returned no output")
        return compact_result(final_output)

    async def add_one(token_ids: list[int], token_labels: list[int]):
        return await engine.add_request(
            f"scheduler-overhead-{uuid.uuid4().hex}",
            {"prompt_token_ids": token_ids},
            build_params(token_labels),
            priority=1000,
        )

    async def collect_one(queue) -> tuple[float, int]:
        final_output = None
        finished = False
        while not finished:
            output = queue.get_nowait() or await queue.get()
            finished = bool(output.finished)
            final_output = output
        if final_output is None:
            raise RuntimeError("scheduled request returned no output")
        return compact_result(final_output)

    if admission == "standard":
        rows = await asyncio.gather(
            *(
                score_one(ids, row_labels)
                for ids, row_labels in zip(token_groups, labels)
            )
        )
    elif admission == "paused_batch":
        await engine.pause_generation(mode="keep", clear_cache=False)
        try:
            queues = await asyncio.gather(
                *(
                    add_one(ids, row_labels)
                    for ids, row_labels in zip(token_groups, labels)
                )
            )
        finally:
            await engine.resume_generation()
        rows = await asyncio.gather(*(collect_one(queue) for queue in queues))
    else:
        raise ValueError(f"unknown scheduler admission: {admission!r}")
    return sum(item[0] for item in rows), sum(item[1] for item in rows)


async def timed(call) -> tuple[float, tuple[float, int]]:
    started = time.perf_counter()
    result = await call()
    return time.perf_counter() - started, result


async def run(args: argparse.Namespace) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    configure_opt_tokenizer(tokenizer, args.model)
    batch_pool_size = int(args.batch_pool_size)
    if batch_pool_size <= 0:
        raise ValueError("batch_pool_size must be positive")
    train_rows, _, _ = load_objective_rows(
        "sst2_classification",
        data_seed=int(args.data_seed),
        num_train=max(64, int(args.batch_size) * batch_pool_size),
        num_dev=1,
        num_eval=1,
        shuffle_impl="hf",
    )
    payloads: list[tuple[list[list[int]], list[list[int]]]] = []
    for batch_index in range(batch_pool_size):
        start = batch_index * int(args.batch_size)
        batch = build_objective_batch(
            train_rows[start : start + int(args.batch_size)],
            tokenizer,
            objective_name="sst2_classification",
            max_length=int(args.max_model_len),
        )
        payloads.append(
            (
                [list(row) for row in batch.token_id_groups]
                * int(args.repeat_sides),
                [list(row) for row in batch.token_labels]
                * int(args.repeat_sides),
            )
        )
    request_counts = [len(token_groups) for token_groups, _ in payloads]
    prompt_token_counts = [
        sum(len(row) for row in token_groups) for token_groups, _ in payloads
    ]
    max_num_seqs = int(args.max_num_seqs or max(request_counts))
    if max_num_seqs < max(request_counts):
        raise ValueError("max_num_seqs must cover the full comparison batch")

    engine_args = AsyncEngineArgs(
        model=args.model,
        enforce_eager=bool(int(args.enforce_eager)),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_batched_tokens=int(args.max_num_batched_tokens),
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=False,
        scheduling_policy="priority",
        disable_log_stats=True,
    )
    init_started = time.perf_counter()
    engine = AsyncLLM.from_engine_args(engine_args)
    init_s = time.perf_counter() - init_started
    direct_s: list[float] = []
    scheduled_s: list[float] = []
    direct_result: tuple[float, int] | None = None
    scheduled_result: tuple[float, int] | None = None
    try:
        total_rounds = int(args.warmup_steps) + int(args.steps)
        # Direct scoring mutates offline model-runner buffers that the scheduler
        # subsequently owns. Measure the scheduler first and never re-enter it
        # after the first direct call.
        for mode in ("scheduled", "direct"):
            for round_index in range(total_rounds):
                token_groups, labels = payloads[round_index % len(payloads)]
                if mode == "scheduled":
                    elapsed, scheduled_result = await timed(
                        lambda: score_scheduled(
                            engine,
                            token_groups,
                            labels,
                            admission=args.scheduler_admission,
                        )
                    )
                    if round_index >= int(args.warmup_steps):
                        scheduled_s.append(elapsed)
                else:
                    elapsed, direct_result = await timed(
                        lambda: score_direct(
                            engine,
                            token_groups,
                            labels,
                            max_logits_tokens=int(args.max_logits_tokens),
                        )
                    )
                    if round_index >= int(args.warmup_steps):
                        direct_s.append(elapsed)
                measured = round_index + 1 - int(args.warmup_steps)
                if measured > 0 and measured % 10 == 0:
                    values = scheduled_s if mode == "scheduled" else direct_s
                    print(
                        f"[scheduler-overhead] mode={mode} step={measured} "
                        f"score_s={values[-1]:.6f}",
                        flush=True,
                    )
    finally:
        engine.shutdown()

    if direct_result is None or scheduled_result is None:
        raise RuntimeError("benchmark produced no score results")
    direct_summary = summarize(direct_s)
    scheduled_summary = summarize(scheduled_s)
    direct_mean = float(direct_summary["mean"])
    scheduled_mean = float(scheduled_summary["mean"])
    return {
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "num_requests": {
                "min": min(request_counts),
                "max": max(request_counts),
            },
            "num_prompt_tokens": {
                "min": min(prompt_token_counts),
                "mean": statistics.fmean(prompt_token_counts),
                "max": max(prompt_token_counts),
            },
            "max_num_seqs_resolved": max_num_seqs,
            "prefix_caching": False,
            "order": "scheduled_then_direct",
        },
        "init_s": init_s,
        "direct_s": direct_summary,
        "scheduled_s": scheduled_summary,
        "scheduled_minus_direct_s": scheduled_mean - direct_mean,
        "scheduled_over_direct": scheduled_mean / direct_mean,
        "score": {
            "direct_nll_sum": direct_result[0],
            "scheduled_nll_sum": scheduled_result[0],
            "nll_sum_abs_diff": abs(direct_result[0] - scheduled_result[0]),
            "direct_num_tokens": direct_result[1],
            "scheduled_num_tokens": scheduled_result[1],
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.repeat_sides <= 0:
        raise SystemExit("batch_size and repeat_sides must be positive")
    if args.steps <= 0 or args.warmup_steps < 0:
        raise SystemExit("steps must be positive and warmup_steps non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[scheduler-overhead-config] "
        + json.dumps({**vars(args), "output_dir": str(args.output_dir)}, sort_keys=True),
        flush=True,
    )
    result = asyncio.run(run(args))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"scheduler_overhead_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "[scheduler-overhead-result] "
        f"json={output_path} direct_mean_s={result['direct_s']['mean']:.6f} "
        f"scheduled_mean_s={result['scheduled_s']['mean']:.6f} "
        f"ratio={result['scheduled_over_direct']:.4f} "
        f"delta_ms={result['scheduled_minus_direct_s'] * 1000.0:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
