import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from zo_vllm.experiment.env import configure_hf_cache, configure_vllm_training_env

configure_hf_cache(PROJECT_ROOT)
configure_vllm_training_env()

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from zo_vllm.core.direction_digest import digest_named_uv
from zo_vllm.core.direct_worker_scorer import (
    eval_sst2_accuracy_direct_worker,
    score_prompt_nll_plus_minus_direct_worker,
    score_sst2_classification_direct_worker,
    score_sst2_classification_plus_minus_direct_worker,
)
from zo_vllm.core.lozo_controller import LOZOConfig, LOZOController
from zo_vllm.core.memory_lora_loader import install_mocks
from zo_vllm.core.temp_lora_runtime import TempLoRARuntime
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.experiment.batching import make_batches
from zo_vllm.experiment.sst2_official import (
    configure_opt_tokenizer,
    sample_sst2_train_dev,
    sample_sst2_validation,
)
from zo_vllm.experiment.stats import summarize, summarize_tail


class RowDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def prepare_prompt_nll_sst2_data(seed, num_samples=1000):
    np.random.seed(seed)
    dataset = load_dataset("glue", "sst2", split="train")
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)
    return PromptDataset([f"{item['sentence']} It was" for item in dataset])


def tokenize_prompts(tokenizer, prompts):
    return [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]


@contextmanager
def nvtx_range(name, enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--direction-scale",
        type=float,
        default=1.0,
        help="Scale applied to U @ V.T directions; Phase 6 uses 1/sqrt(rank).",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--step-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-dev", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--train-sampler", choices=["sequential", "hf_random"], default="sequential")
    parser.add_argument("--dataloader-seed", type=int, default=None)
    parser.add_argument(
        "--train-objective",
        choices=["sst2_classification", "prompt_nll"],
        default="sst2_classification",
    )
    parser.add_argument("--zo-random-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--direction-sampling", choices=["exact", "flat"], default="exact")
    parser.add_argument("--train-scope", choices=["lora_only"], default="lora_only")
    parser.add_argument("--batch-invariant", choices=["0", "1"], default="0")
    parser.add_argument("--enforce-eager", choices=["0", "1"], default="0")
    parser.add_argument("--lora-residency", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--lora-injection", choices=["auto", "direct", "manager"], default="auto")
    parser.add_argument("--weight-update", choices=["copy", "direct"], default="direct")
    parser.add_argument("--weight-update-precision", choices=["float32", "param"], default="param")
    parser.add_argument("--direct-update-mode", choices=["immediate", "accumulate"], default="immediate")
    parser.add_argument("--qkv-weight-update", choices=["separate", "batched"], default="separate")
    parser.add_argument("--sync-weight-update", choices=["0", "1"], default="1")
    parser.add_argument("--scoring-backend", choices=["generate", "direct_worker"], default="generate")
    parser.add_argument("--direct-worker-max-logits-tokens", type=int, default=8192)
    parser.add_argument("--direct-worker-loss-impl", choices=["logprobs", "cross_entropy"], default="logprobs")
    parser.add_argument("--slot-pipeline", choices=["0", "1"], default="0")
    parser.add_argument("--base-eval-mode", choices=["generate", "direct_worker", "skip"], default="generate")
    parser.add_argument("--profile-mode", choices=["minimal", "detailed"], default="minimal")
    parser.add_argument("--fuse-lora-score", choices=["0", "1"], default="0")
    parser.add_argument("--direction-digest", action="store_true")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--trace-step-events", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--train-loss-interval", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--direct-controller-source",
        choices=["vllm_metadata", "hf_master"],
        default="vllm_metadata",
    )
    parser.add_argument("--model-name", default="facebook/opt-2.7b")
    parser.add_argument("--direct-lora-from-directions", choices=["0", "1"], default="0")
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-accuracy-samples", type=int, default=512)
    parser.add_argument("--accuracy-eval-mode", choices=["auto", "full", "skip"], default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    data_seed = args.seed if args.data_seed is None else args.data_seed
    accuracy_eval_mode = args.accuracy_eval_mode
    if accuracy_eval_mode == "auto":
        accuracy_eval_mode = "skip" if args.base_eval_mode == "skip" else "full"

    os.environ["VLLM_BATCH_INVARIANT"] = args.batch_invariant
    if args.lora_residency == "gpu" and not torch.cuda.is_available():
        raise SystemExit("--lora-residency gpu requires CUDA")
    lora_injection = args.lora_injection
    if lora_injection == "auto":
        lora_injection = "direct" if args.lora_residency == "gpu" else "manager"
    if args.lora_residency == "cpu":
        if lora_injection != "manager":
            raise SystemExit("--lora-injection direct requires --lora-residency gpu")
        install_mocks()
    if args.scoring_backend == "direct_worker" and (
        args.lora_residency != "gpu" or lora_injection != "direct"
    ):
        raise SystemExit(
            "--scoring-backend direct_worker requires GPU residency and direct LoRA injection"
        )
    slot_pipeline = bool(int(args.slot_pipeline))
    fuse_lora_score = bool(int(args.fuse_lora_score))
    if slot_pipeline and args.scoring_backend != "direct_worker":
        raise SystemExit("--slot-pipeline requires --scoring-backend direct_worker")
    direct_lora_from_directions = bool(int(args.direct_lora_from_directions))
    if direct_lora_from_directions and (
        args.lora_residency != "gpu" or lora_injection != "direct"
    ):
        raise SystemExit(
            "--direct-lora-from-directions requires GPU residency and direct LoRA injection"
        )
    if fuse_lora_score and (
        args.scoring_backend != "direct_worker"
        or not direct_lora_from_directions
        or slot_pipeline
    ):
        raise SystemExit(
            "--fuse-lora-score requires direct_worker, "
            "--direct-lora-from-directions 1, and --slot-pipeline 0"
        )
    if fuse_lora_score:
        raise SystemExit(
            "--fuse-lora-score is not supported for SST-2 classification loss alignment"
        )
    use_accumulated_update = args.direct_update_mode == "accumulate"
    if use_accumulated_update:
        if args.weight_update != "direct":
            raise SystemExit("--direct-update-mode accumulate requires --weight-update direct")
        if args.weight_update_precision != "param":
            raise SystemExit("--direct-update-mode accumulate requires --weight-update-precision param")
        if not direct_lora_from_directions:
            raise SystemExit("--direct-update-mode accumulate requires --direct-lora-from-directions 1")
        if slot_pipeline:
            raise SystemExit("--direct-update-mode accumulate does not support --slot-pipeline 1")
    step_nvtx_enabled = os.environ.get("VLLM_ZO_STEP_NVTX", "0") == "1"
    step_nvtx_skip = int(os.environ.get("VLLM_ZO_STEP_NVTX_SKIP", "0"))
    step_nvtx_limit = int(os.environ.get("VLLM_ZO_STEP_NVTX_LIMIT", "0"))

    def should_emit_step_nvtx(measured_index):
        if not step_nvtx_enabled:
            return False
        if measured_index <= step_nvtx_skip:
            return False
        return step_nvtx_limit <= 0 or measured_index <= step_nvtx_skip + step_nvtx_limit

    from vllm import LLM

    model_name = args.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "phase3", "results", timestamp, "vllm"
    )
    os.makedirs(output_dir, exist_ok=True)
    if args.num_samples < args.batch_size:
        raise SystemExit("--num-samples must be greater than or equal to --batch-size")

    print(
        f"[vLLM] steps={args.steps} warmup_steps={args.warmup_steps} "
        f"batch_size={args.batch_size} rank={args.rank} "
        f"lr={args.lr} eps={args.eps} direction_scale={args.direction_scale} "
        f"profile_mode={args.profile_mode} "
        f"enforce_eager={args.enforce_eager} scoring_backend={args.scoring_backend} "
        f"sync_weight_update={args.sync_weight_update} slot_pipeline={args.slot_pipeline} "
        f"qkv_weight_update={args.qkv_weight_update} "
        f"direct_update_mode={args.direct_update_mode} "
        f"fuse_lora_score={args.fuse_lora_score} "
        f"direct_worker_loss_impl={args.direct_worker_loss_impl} "
        f"direct_lora_from_directions={args.direct_lora_from_directions} "
        f"direction_sampling={args.direction_sampling} "
        f"train_objective={args.train_objective} "
        f"step_nvtx={int(step_nvtx_enabled)} "
        f"model_name={model_name} seed={args.seed} data_seed={data_seed}",
        flush=True,
    )

    model_config = AutoConfig.from_pretrained(model_name)
    hf_model = None
    if args.weight_update == "copy" or args.direct_controller_source == "hf_master":
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    configure_opt_tokenizer(tokenizer, model_name)
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": bool(int(args.enforce_eager)),
        "enable_lora": True,
        "max_lora_rank": args.rank,
        "max_loras": 4 if slot_pipeline else 2,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.kv_cache_memory_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(
        **llm_kwargs,
    )
    weight_sync = WeightSync(llm, num_layers=model_config.num_hidden_layers)

    config = LOZOConfig(
        rank=args.rank,
        eps=args.eps,
        lr=args.lr,
        step_interval=args.step_interval,
        master_device="cuda" if torch.cuda.is_available() else "cpu",
        random_device=args.zo_random_device,
        train_scope=args.train_scope,
        direction_sampling=args.direction_sampling,
        direction_scale=args.direction_scale,
    )
    param_metadata = None
    if args.weight_update == "direct" and args.direct_controller_source == "vllm_metadata":
        param_metadata = weight_sync.get_hf_param_metadata()
    controller = LOZOController(hf_model, config, param_metadata=param_metadata)
    temp_loras = [
        TempLoRARuntime(
            rank=args.rank,
            num_layers=model_config.num_hidden_layers,
            plus_id=9001,
            minus_id=9002,
            residency=args.lora_residency,
            injection=lora_injection,
            llm=llm,
            base_model_name=model_name,
            hidden_size=model_config.hidden_size,
            ffn_dim=model_config.ffn_dim,
        )
    ]
    if slot_pipeline:
        temp_loras.append(
            TempLoRARuntime(
                rank=args.rank,
                num_layers=model_config.num_hidden_layers,
                plus_id=9003,
                minus_id=9004,
                residency=args.lora_residency,
                injection=lora_injection,
                llm=llm,
                base_model_name=model_name,
                hidden_size=model_config.hidden_size,
                ffn_dim=model_config.ffn_dim,
            )
        )
    for runtime in temp_loras:
        runtime.register_slots()
    temp_lora = temp_loras[0]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.train_objective == "prompt_nll":
        dataset = prepare_prompt_nll_sst2_data(data_seed, num_samples=args.num_samples)
        dev_rows = []
        valid_rows_cls = []
        train_items = tokenize_prompts(tokenizer, list(dataset.prompts))
    else:
        train_rows, dev_rows = sample_sst2_train_dev(
            seed=data_seed,
            num_train=args.num_samples,
            num_dev=args.num_dev,
        )
        valid_rows_cls = sample_sst2_validation(
            seed=data_seed,
            num_eval=args.eval_accuracy_samples,
        )
        dataset = RowDataset(train_rows)
        train_items = list(dataset.rows)
    dataloader_seed = args.seed if args.dataloader_seed is None else args.dataloader_seed
    row_batches = make_batches(
        train_items,
        args.batch_size,
        sampler=args.train_sampler,
        seed=dataloader_seed,
    )
    if not row_batches:
        raise SystemExit("no full training batches; increase --num-samples or lower --batch-size")
    if args.base_eval_mode == "skip":
        initial_loss = None
        print("[vLLM] initial_loss=skipped", flush=True)
    elif args.train_objective == "sst2_classification":
        initial_loss = score_sst2_classification_direct_worker(
            llm,
            dev_rows,
            tokenizer,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
        )
        print(f"[vLLM] initial_loss={initial_loss:.6f}", flush=True)
    else:
        initial_loss = None
        print("[vLLM] initial_loss=skipped", flush=True)

    np.random.seed(args.seed)
    eval_losses = [] if initial_loss is None else [{"step": 0, "loss": float(initial_loss)}]
    if accuracy_eval_mode == "skip":
        initial_dev_acc = None
        initial_valid_acc = None
        print("[vLLM] initial_acc=skipped", flush=True)
    elif args.train_objective == "sst2_classification":
        initial_dev_acc = eval_sst2_accuracy_direct_worker(
            llm,
            tokenizer,
            dev_rows,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
        )
        initial_valid_acc = eval_sst2_accuracy_direct_worker(
            llm,
            tokenizer,
            valid_rows_cls,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
        )
        if initial_dev_acc is not None:
            print(f"[vLLM] initial_acc={initial_dev_acc:.6f}", flush=True)
        if initial_valid_acc is not None:
            print(f"[vLLM] initial_valid_acc={initial_valid_acc:.6f}", flush=True)
    else:
        initial_dev_acc = None
        initial_valid_acc = None
        print("[vLLM] initial_acc=skipped", flush=True)
    eval_metrics = [] if initial_loss is None else [{
        "step": 0,
        "loss": float(initial_loss),
        "accuracy": initial_dev_acc,
    }]
    history = []
    timing = {
        "step_s": [],
        "direction_s": [],
        "build_lora_s": [],
        "lora_update_s": [],
        "score_s": [],
        "weight_update_s": [],
        "score_request_build_s": [],
        "score_generate_s": [],
        "score_postprocess_s": [],
        "score_direct_worker_s": [],
        "prep_launch_s": [],
        "prep_wait_s": [],
        "weight_fold_s": [],
    }

    prep_stream = torch.cuda.Stream() if slot_pipeline and torch.cuda.is_available() else None
    ckpt_root = os.path.join(output_dir, "checkpoints")
    if args.save_interval > 0:
        os.makedirs(ckpt_root, exist_ok=True)
    ckpt_paths = []
    accumulated_u: dict[str, torch.Tensor] = {}

    def attach_accumulated_lora(directions_2d):
        if not use_accumulated_update:
            return
        for name, direction in directions_2d.items():
            U = direction["U"]
            acc = accumulated_u.get(name)
            if acc is None:
                acc = torch.zeros_like(U)
                accumulated_u[name] = acc
            direction["U_accum"] = acc

    def accumulate_step_update(directions_2d, c: float) -> None:
        if not use_accumulated_update:
            return
        for name, direction in directions_2d.items():
            acc = accumulated_u.get(name)
            if acc is None:
                acc = torch.zeros_like(direction["U"])
                accumulated_u[name] = acc
            scale = float(direction.get("scale", config.direction_scale))
            acc.add_(direction["U"], alpha=-config.lr * c * scale)

    def fold_accumulated_update_if_needed(force: bool = False) -> float:
        if not use_accumulated_update or not accumulated_u:
            return 0.0
        if not force and (controller.step <= 0 or controller.step % config.step_interval != 0):
            return 0.0
        fold_directions = {}
        for name, acc in accumulated_u.items():
            V = controller.v_cache.get(name)
            if V is None:
                continue
            fold_directions[name] = {
                "U": acc,
                "V": V,
                "V_T": controller.vt_cache.get(name),
            }
        if not fold_directions:
            accumulated_u.clear()
            return 0.0
        t0 = time.perf_counter()
        weight_sync.apply_lozo_update(
            fold_directions,
            {},
            c=-1.0,
            lr=1.0,
            weight_decay=0.0,
            precision=args.weight_update_precision,
            sync_device=bool(int(args.sync_weight_update)),
            qkv_update_mode=args.qkv_weight_update,
        )
        for acc in accumulated_u.values():
            acc.zero_()
        return time.perf_counter() - t0

    def save_runtime_checkpoint(measured_index, latest_eval_loss):
        if args.save_interval <= 0:
            return
        ckpt = {
            "step": int(measured_index),
            "eval_loss": None if latest_eval_loss is None else float(latest_eval_loss),
            "timing_counts": {k: len(v) for k, v in timing.items()},
            "history_count": len(history),
            "timestamp": datetime.now().isoformat(),
        }
        ckpt_path = os.path.join(ckpt_root, f"step_{measured_index:07d}.json")
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        ckpt_paths.append(ckpt_path)
        if args.save_total_limit > 0 and len(ckpt_paths) > args.save_total_limit:
            stale = ckpt_paths.pop(0)
            if os.path.exists(stale):
                os.remove(stale)
        print(f"[vLLM] checkpoint_saved={ckpt_path}", flush=True)

    def prepare_lora_state(runtime, slot_idx, step_value, random_seed):
        prep_t0 = time.perf_counter()
        stream_context = torch.cuda.stream(prep_stream) if prep_stream is not None else nullcontext()
        with stream_context:
            t0 = time.perf_counter()
            directions_2d, directions_1d = controller.sample_direction(random_seed)
            direction_digest = None
            if args.direction_digest:
                direction_digest = digest_named_uv(
                    (name, item["U"], item["V"]) for name, item in directions_2d.items()
                )
            direction_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            if direct_lora_from_directions:
                build_lora_s = 0.0
                if slot_pipeline:
                    # Each double-buffered slot pair must receive A/V at least
                    # once for the current V cache generation. The simplest
                    # safe training path writes A together with B for pipeline
                    # mode; this is still cheap enough to overlap with scoring.
                    for direction in directions_2d.values():
                        direction["v_refreshed"] = True
                runtime.update_plus_minus_from_directions(
                    directions_2d,
                    eps=config.eps * config.direction_scale,
                    step=step_value,
                )
            else:
                lora_device = "cuda" if args.lora_residency == "gpu" else "cpu"
                plus_A, plus_B, minus_A, minus_B = controller.build_temp_lora_pair_tensors(
                    directions_2d, output_device=lora_device
                )
                build_lora_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                runtime.update_plus_minus(plus_A, plus_B, minus_A, minus_B, step=step_value)
            lora_update_s = time.perf_counter() - t0

            ready_event = None
            if prep_stream is not None:
                ready_event = torch.cuda.Event()
                ready_event.record(prep_stream)

        return {
            "runtime": runtime,
            "slot_idx": slot_idx,
            "random_seed": int(random_seed),
            "directions_2d": directions_2d,
            "directions_1d": directions_1d,
            "direction_digest": direction_digest,
            "direction_s": direction_s,
            "build_lora_s": build_lora_s,
            "lora_update_s": lora_update_s,
            "prep_launch_s": time.perf_counter() - prep_t0,
            "ready_event": ready_event,
        }

    total_target_steps = args.steps + args.warmup_steps
    step = 0
    measured_train_t0 = None
    measured_train_t1 = None
    last_progress_time = None
    last_progress_step = 0
    train_loss_window_sum = 0.0
    train_loss_window_count = 0
    prep_executor = ThreadPoolExecutor(max_workers=1) if slot_pipeline else None
    pipelined_future = None
    if slot_pipeline:
        pipelined_future = prep_executor.submit(
            prepare_lora_state,
            temp_loras[0],
            0,
            1,
            np.random.randint(1000000000),
        )
    try:
        for _epoch in range(total_target_steps // len(row_batches) + 2):
            for batch_rows in row_batches:
                if step >= total_target_steps:
                    break
                step += 1
                measured_step = step > args.warmup_steps
                measured_index = step - args.warmup_steps
                if measured_step and measured_train_t0 is None:
                    measured_train_t0 = time.perf_counter()

                def record_timing(key, value):
                    if measured_step:
                        timing[key].append(value)

                step_t0 = time.perf_counter()
                emit_step_nvtx = measured_step and should_emit_step_nvtx(measured_index)
                step_total_nvtx_pushed = emit_step_nvtx and torch.cuda.is_available()
                if step_total_nvtx_pushed:
                    torch.cuda.nvtx.range_push("zo_step.total")
                if slot_pipeline:
                    if pipelined_future is None:
                        raise RuntimeError("slot pipeline state was not submitted")
                    wait_t0 = time.perf_counter()
                    pipelined_state = pipelined_future.result()
                    record_timing("prep_wait_s", time.perf_counter() - wait_t0)
                    if pipelined_state["ready_event"] is not None:
                        torch.cuda.current_stream().wait_event(pipelined_state["ready_event"])
                    temp_lora = pipelined_state["runtime"]
                    random_seed = pipelined_state["random_seed"]
                    directions_2d = pipelined_state["directions_2d"]
                    directions_1d = pipelined_state["directions_1d"]
                    direction_digest = pipelined_state["direction_digest"]
                    record_timing("direction_s", pipelined_state["direction_s"])
                    record_timing("build_lora_s", pipelined_state["build_lora_s"])
                    record_timing("lora_update_s", pipelined_state["lora_update_s"])
                    record_timing("prep_launch_s", pipelined_state["prep_launch_s"])
                    if args.profile_mode == "detailed" and measured_step:
                        for worker_info in temp_lora.last_update_info.get("workers", []):
                            for key, value in worker_info.get("profile_s", {}).items():
                                timing.setdefault(f"lora_worker_{key}_s", []).append(
                                    float(value)
                                )
                    pipelined_future = None
                    if step < total_target_steps:
                        next_slot_idx = 1 - int(pipelined_state["slot_idx"])
                        pipelined_future = prep_executor.submit(
                            prepare_lora_state,
                            temp_loras[next_slot_idx],
                            next_slot_idx,
                            step + 1,
                            np.random.randint(1000000000),
                        )
                else:
                    record_timing("prep_wait_s", 0.0)
                    random_seed = np.random.randint(1000000000)

                    if args.trace_step_events:
                        print(f"[trace] step={step} direction_start", flush=True)
                    t0 = time.perf_counter()
                    with nvtx_range("zo_step.direction", emit_step_nvtx):
                        fold_s = fold_accumulated_update_if_needed()
                        directions_2d, directions_1d = controller.sample_direction(random_seed)
                        direction_digest = None
                        if args.direction_digest:
                            direction_digest = digest_named_uv(
                                (name, item["U"], item["V"]) for name, item in directions_2d.items()
                            )
                    record_timing("weight_fold_s", fold_s)
                    record_timing("direction_s", time.perf_counter() - t0 - fold_s)
                    if args.trace_step_events:
                        print(f"[trace] step={step} direction_done", flush=True)

                    t0 = time.perf_counter()
                    if direct_lora_from_directions:
                        record_timing("build_lora_s", 0.0)
                        if args.trace_step_events:
                            print(f"[trace] step={step} lora_update_from_directions_start", flush=True)
                        with nvtx_range("zo_step.lora_update", emit_step_nvtx):
                            attach_accumulated_lora(directions_2d)
                            temp_lora.update_plus_minus_from_directions(
                                directions_2d,
                                eps=config.eps * config.direction_scale,
                                step=step,
                            )
                    else:
                        if args.trace_step_events:
                            print(f"[trace] step={step} build_lora_start", flush=True)
                        lora_device = "cuda" if args.lora_residency == "gpu" else "cpu"
                        with nvtx_range("zo_step.build_lora", emit_step_nvtx):
                            plus_A, plus_B, minus_A, minus_B = controller.build_temp_lora_pair_tensors(
                                directions_2d, output_device=lora_device
                            )
                        record_timing("build_lora_s", time.perf_counter() - t0)
                        t0 = time.perf_counter()
                        if args.trace_step_events:
                            print(f"[trace] step={step} lora_update_start", flush=True)
                        with nvtx_range("zo_step.lora_update", emit_step_nvtx):
                            temp_lora.update_plus_minus(plus_A, plus_B, minus_A, minus_B, step=step)
                    record_timing("lora_update_s", time.perf_counter() - t0)
                    if args.profile_mode == "detailed" and measured_step:
                        for worker_info in temp_lora.last_update_info.get("workers", []):
                            for key, value in worker_info.get("profile_s", {}).items():
                                timing.setdefault(f"lora_worker_{key}_s", []).append(
                                    float(value)
                                )
                    if args.trace_step_events:
                        print(f"[trace] step={step} lora_update_done", flush=True)
                    record_timing("prep_launch_s", 0.0)

                if args.trace_step_events:
                    print(f"[trace] step={step} score_start", flush=True)
                t0 = time.perf_counter()
                if emit_step_nvtx and torch.cuda.is_available():
                    torch.cuda.nvtx.range_push("zo_step.score")
                if args.scoring_backend == "direct_worker":
                    if args.train_objective == "prompt_nll":
                        loss_plus, loss_minus, score_detail = (
                            score_prompt_nll_plus_minus_direct_worker(
                                llm,
                                batch_rows,
                                plus_id=temp_lora.plus_id,
                                minus_id=temp_lora.minus_id,
                                max_logits_tokens=args.direct_worker_max_logits_tokens,
                                loss_impl=args.direct_worker_loss_impl,
                            )
                        )
                    else:
                        loss_plus, loss_minus, score_detail = (
                            score_sst2_classification_plus_minus_direct_worker(
                                llm,
                                batch_rows,
                                tokenizer,
                                plus_id=temp_lora.plus_id,
                                minus_id=temp_lora.minus_id,
                                max_logits_tokens=args.direct_worker_max_logits_tokens,
                                loss_impl=args.direct_worker_loss_impl,
                            )
                        )
                    if args.profile_mode == "detailed" and measured_step:
                        timing["score_direct_worker_s"].append(
                            score_detail["score_direct_worker_s"]
                        )
                        for key, value in score_detail.items():
                            if key.startswith("score_worker_"):
                                timing.setdefault(key, []).append(float(value))
                elif args.profile_mode == "detailed":
                    raise SystemExit("SST-2 classification loss requires --scoring-backend direct_worker")
                else:
                    raise SystemExit("SST-2 classification loss requires --scoring-backend direct_worker")
                if emit_step_nvtx and torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
                record_timing("score_s", time.perf_counter() - t0)
                if args.trace_step_events:
                    print(f"[trace] step={step} score_done", flush=True)

                c = controller.compute_c(loss_plus, loss_minus)

                if args.trace_step_events:
                    print(f"[trace] step={step} weight_update_start", flush=True)
                t0 = time.perf_counter()
                with nvtx_range("zo_step.weight_update", emit_step_nvtx):
                    if args.weight_update == "copy":
                        updated_weights = controller.apply_update_to_master(
                            directions_2d, directions_1d, c
                        )
                        weight_sync.sync(updated_weights)
                    elif use_accumulated_update:
                        accumulate_step_update(directions_2d, c)
                    else:
                        weight_sync.apply_lozo_update(
                            directions_2d,
                            directions_1d,
                            c=c,
                            lr=config.lr,
                            weight_decay=config.weight_decay,
                            precision=args.weight_update_precision,
                            sync_device=bool(int(args.sync_weight_update)),
                            qkv_update_mode=args.qkv_weight_update,
                        )
                record_timing("weight_update_s", time.perf_counter() - t0)
                if args.profile_mode == "detailed" and measured_step:
                    for worker_info in weight_sync.last_update_info.get("workers", []):
                        if not worker_info:
                            continue
                        for key, value in worker_info.get("profile_s", {}).items():
                            timing.setdefault(f"weight_worker_{key}_s", []).append(
                                float(value)
                            )
                if args.trace_step_events:
                    print(f"[trace] step={step} weight_update_done", flush=True)
                if step_total_nvtx_pushed:
                    torch.cuda.nvtx.range_pop()
                if measured_step:
                    timing["step_s"].append(time.perf_counter() - step_t0)
                    measured_train_t1 = time.perf_counter()

                    if args.record_history:
                        history.append({
                            "step": measured_index,
                            "raw_step": step,
                            "seed": int(random_seed),
                            "loss_plus": float(loss_plus),
                            "loss_minus": float(loss_minus),
                            "c": float(c),
                            "direction_digest": direction_digest,
                            "step_s": float(timing["step_s"][-1]),
                        })

                    if args.train_loss_interval > 0:
                        train_loss_window_sum += float(loss_plus)
                        train_loss_window_count += 1
                        if measured_index % args.train_loss_interval == 0:
                            train_loss = train_loss_window_sum / max(
                                train_loss_window_count, 1
                            )
                            print(
                                f"[vLLM] step={measured_index} "
                                f"train_loss={train_loss:.6f} "
                                f"learning_rate={args.lr:.6g}",
                                flush=True,
                            )
                            train_loss_window_sum = 0.0
                            train_loss_window_count = 0

                    if args.progress_interval > 0 and measured_index % args.progress_interval == 0:
                        progress_now = time.perf_counter()
                        cumulative_s = (
                            0.0
                            if measured_train_t0 is None
                            else progress_now - measured_train_t0
                        )
                        window_s = (
                            cumulative_s
                            if last_progress_time is None
                            else progress_now - last_progress_time
                        )
                        window_steps = measured_index - last_progress_step
                        print(
                            f"[vLLM] step={measured_index} seed={int(random_seed)} "
                            f"plus={loss_plus:.6f} minus={loss_minus:.6f} "
                            f"c={c:.6f} step_s={timing['step_s'][-1]:.6f} "
                            f"window_steps={window_steps} window_s={window_s:.6f} "
                            f"cumulative_s={cumulative_s:.6f}",
                            flush=True,
                        )
                        last_progress_time = progress_now
                        last_progress_step = measured_index

                    if args.eval_interval > 0 and measured_index % args.eval_interval == 0:
                        if args.base_eval_mode == "skip":
                            print(f"[vLLM] step={measured_index} eval_loss=skipped", flush=True)
                            save_runtime_checkpoint(measured_index, None)
                        elif args.train_objective == "sst2_classification":
                            eval_fold_s = fold_accumulated_update_if_needed(force=True)
                            if eval_fold_s:
                                print(
                                    f"[vLLM] step={measured_index} "
                                    f"eval_weight_fold_s={eval_fold_s:.6f}",
                                    flush=True,
                                )
                            val_loss = score_sst2_classification_direct_worker(
                                llm,
                                dev_rows,
                                tokenizer,
                                max_logits_tokens=args.direct_worker_max_logits_tokens,
                                loss_impl=args.direct_worker_loss_impl,
                            )
                            eval_losses.append({"step": measured_index, "loss": float(val_loss)})
                            val_acc = eval_sst2_accuracy_direct_worker(
                                llm,
                                tokenizer,
                                dev_rows,
                                max_logits_tokens=args.direct_worker_max_logits_tokens,
                                loss_impl=args.direct_worker_loss_impl,
                            )
                            eval_metrics.append(
                                {"step": measured_index, "loss": float(val_loss), "accuracy": val_acc}
                            )
                            print(f"[vLLM] step={measured_index} eval_loss={val_loss:.6f}", flush=True)
                            if val_acc is not None:
                                print(f"[vLLM] step={measured_index} eval_acc={val_acc:.6f}", flush=True)
                            save_runtime_checkpoint(measured_index, val_loss)
                        else:
                            print(f"[vLLM] step={measured_index} eval_loss=skipped", flush=True)
                            save_runtime_checkpoint(measured_index, None)
            if step >= total_target_steps:
                break
    finally:
        if prep_executor is not None:
            prep_executor.shutdown(wait=True)

    total_s = 0.0 if measured_train_t0 is None else measured_train_t1 - measured_train_t0
    final_fold_s = fold_accumulated_update_if_needed(force=True)
    if final_fold_s:
        print(f"[vLLM] final_weight_fold_s={final_fold_s:.6f}", flush=True)
    if args.base_eval_mode == "skip":
        final_loss = None
        print("[vLLM] final_loss=skipped", flush=True)
    elif args.train_objective == "sst2_classification":
        final_loss = score_sst2_classification_direct_worker(
            llm,
            dev_rows,
            tokenizer,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
        )
        if not eval_losses or eval_losses[-1]["step"] != args.steps:
            eval_losses.append({"step": args.steps, "loss": float(final_loss)})
        print(f"[vLLM] final_loss={final_loss:.6f}", flush=True)
    else:
        final_loss = None
        print("[vLLM] final_loss=skipped", flush=True)
    final_dev_acc = None
    final_valid_acc = None
    if accuracy_eval_mode == "skip":
        print("[vLLM] final_acc=skipped", flush=True)
    elif args.train_objective == "sst2_classification":
        final_dev_acc = eval_sst2_accuracy_direct_worker(
            llm,
            tokenizer,
            dev_rows,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
        )
        final_valid_acc = eval_sst2_accuracy_direct_worker(
            llm,
            tokenizer,
            valid_rows_cls,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
        )
    else:
        print("[vLLM] final_acc=skipped", flush=True)
    if not eval_metrics or eval_metrics[-1]["step"] != args.steps:
        eval_metrics.append(
            {
                "step": args.steps,
                "loss": None if final_loss is None else float(final_loss),
                "accuracy": final_dev_acc,
            }
        )
    if final_dev_acc is not None:
        print(f"[vLLM] final_dev_acc={final_dev_acc:.6f}", flush=True)
    if final_valid_acc is not None:
        print(f"[vLLM] final_acc={final_valid_acc:.6f}", flush=True)

    output_file = os.path.join(output_dir, f"vllm_perf_{args.profile_mode}_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": vars(args) | {
                "model": model_name,
                "backend": "vllm",
                "lora_injection_resolved": lora_injection,
                "accuracy_eval_mode_resolved": accuracy_eval_mode,
                "train_objective_resolved": args.train_objective,
            },
            "initial_loss": None if initial_loss is None else float(initial_loss),
            "final_loss": None if final_loss is None else float(final_loss),
            "loss_change": (
                None
                if initial_loss is None or final_loss is None
                else float(final_loss - initial_loss)
            ),
            "eval_losses": eval_losses,
            "eval_metrics": eval_metrics,
            "eval_metrics_accuracy_scope": "dev",
            "final_accuracy_scope": "validation",
            "initial_accuracy": initial_dev_acc,
            "initial_dev_accuracy": initial_dev_acc,
            "initial_valid_accuracy": initial_valid_acc,
            "final_accuracy": final_valid_acc,
            "final_dev_accuracy": final_dev_acc,
            "final_valid_accuracy": final_valid_acc,
            "history": history,
            "timing": {
                "total_s": float(total_s),
                **{key: summarize(value) for key, value in timing.items()},
                "tail_100": {
                    key: summarize_tail(value, 100) for key, value in timing.items()
                },
            },
            "checkpoints": ckpt_paths,
        }, f, indent=2)

    for runtime in temp_loras:
        runtime.cleanup()
    print(f"[vLLM] saved={output_file}", flush=True)


if __name__ == "__main__":
    main()
