# ruff: noqa: E402
from contextlib import contextmanager
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, PROJECT_ROOT)

from zo_vllm.experiment.infra.env import configure_hf_cache, configure_vllm_training_env

configure_hf_cache(PROJECT_ROOT)
configure_vllm_training_env()

import numpy as np
import torch
from transformers import AutoConfig

from zo_vllm.config import VLLMZOConfig, ZOVLLMEngineConfig
from zo_vllm.core.lora_scope import (
    lora_scope_includes_embeddings,
    resolve_lora_target_modules,
)
from zo_vllm.core.weight_sync import WeightSync
from zo_vllm.engine import ZOVLLMEngine
from zo_vllm.experiment.infra.batching import (
    make_train_dataloader,
)
from zo_vllm.experiment.runners.args import build_vllm_zo_task_arg_parser
from zo_vllm.experiment.runners.checkpoint_policy import (
    BestMetricTracker,
    resolve_runtime_checkpoint_settings,
)
from zo_vllm.experiment.runners.checkpointing import (
    RuntimeCheckpointManager,
)
from zo_vllm.training.native_checkpoint import (
    clear_update_state_for_loaded_checkpoint,
    load_native_checkpoint_into_workers,
)
from zo_vllm.training.lora_checkpoint import (
    load_lora_bank_checkpoint,
    restore_direction_provider_v_cache_from_lora_bank,
)
from zo_vllm.experiment.runners.intervals import resolve_runner_intervals
from zo_vllm.experiment.runners.trainer_state import (
    load_zo_trainer_state,
    merge_zo_trainer_state,
)
from zo_vllm.training.objective_router import (
    eval_objective_metrics,
    score_current_batch_train_loss,
    score_objective_loss,
    score_periodic_eval,
)
from zo_vllm.experiment.runners.output import (
    resolve_vllm_output_paths,
    write_vllm_perf_json,
)
from zo_vllm.experiment.runners.run_logging import (
    print_interval_summary,
    print_vllm_run_header,
)
from zo_vllm.experiment.runners.runtime_modes import resolve_vllm_zo_runtime_modes
from zo_vllm.experiment.runners.u_snapshot import USnapshotRecorder
from zo_vllm.experiment.runners.wandb_logging import init_wandb_logger
from zo_vllm.training.task_batches import (
    load_objective_rows,
    prepare_prompt_nll_sst2_data,
    resolve_objective_name,
    tokenize_prompts,
)
from zo_vllm.training import (
    VLLMZOModel,
    VLLMZOTrainer,
    VLLMZOTrainerCallback,
    ZOTrainingArguments,
    ZOTaskDataCollator,
    ZOTaskEncodingConfig,
    build_tokenizer,
)
from zo_vllm.training.update_state import AccumulatedLowRankUpdateState
from zo_vllm.training.update_bank_state import BlockLoRAUpdateBankState


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
    args = build_vllm_zo_task_arg_parser().parse_args()
    args.train_objective = resolve_objective_name(
        args.train_objective,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        task_name=args.task_name,
    )
    if args.model_name_or_path:
        args.model_name = args.model_name_or_path
    data_seed = int(args.seed if args.data_seed is None else args.data_seed)
    dataloader_seed = int(
        args.seed if args.dataloader_seed is None else args.dataloader_seed
    )
    args.data_seed = data_seed
    args.dataloader_seed = dataloader_seed
    os.environ["ZO_TASK_SHUFFLE_IMPL"] = args.task_shuffle_impl
    accuracy_eval_mode = args.accuracy_eval_mode
    if accuracy_eval_mode == "auto":
        accuracy_eval_mode = "skip" if args.base_eval_mode == "skip" else "full"

    if not torch.cuda.is_available():
        raise SystemExit("vllm_zo_task requires CUDA for GPU direct LoRA slots")
    runtime_modes = resolve_vllm_zo_runtime_modes(args)
    use_lora_bank_update = runtime_modes.use_lora_bank_update
    use_accumulated_update = runtime_modes.use_accumulated_update
    use_accumulated_lora_eval = runtime_modes.use_accumulated_lora_eval
    effective_direction_scale = runtime_modes.effective_direction_scale
    direction_scale_mode = runtime_modes.direction_scale_mode
    direction_scale_note = runtime_modes.direction_scale_note
    direction_scale_applies_to = runtime_modes.direction_scale_applies_to
    use_unified_stepper = runtime_modes.use_unified_stepper
    perturb_embeddings = lora_scope_includes_embeddings(
        args.train_scope,
        perturb_embeddings=bool(int(args.perturb_embeddings)),
    )
    step_nvtx_enabled = os.environ.get("VLLM_ZO_STEP_NVTX", "0") == "1"
    step_nvtx_skip = int(os.environ.get("VLLM_ZO_STEP_NVTX_SKIP", "0"))
    step_nvtx_limit = int(os.environ.get("VLLM_ZO_STEP_NVTX_LIMIT", "0"))

    def should_emit_step_nvtx(measured_index):
        if not step_nvtx_enabled:
            return False
        if measured_index <= step_nvtx_skip:
            return False
        return (
            step_nvtx_limit <= 0 or measured_index <= step_nvtx_skip + step_nvtx_limit
        )

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    from vllm import LLM

    prequant_model = args.prequant_model or os.environ.get("PHASE7_PREQUANT_MODEL")
    model_name = prequant_model or args.model_name
    lora_slot_rank = runtime_modes.lora_slot_rank
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths = resolve_vllm_output_paths(
        args=args,
        project_root=PROJECT_ROOT,
        model_name=model_name,
        timestamp=timestamp,
    )
    output_dir = output_paths.output_dir
    output_root_resolved = output_paths.output_root
    experiment_name_resolved = output_paths.experiment_name
    os.makedirs(output_dir, exist_ok=True)
    if args.num_samples < args.batch_size:
        raise SystemExit("--num-samples must be greater than or equal to --batch-size")

    print_vllm_run_header(
        args=args,
        runtime_modes=runtime_modes,
        model_name=model_name,
        prequant_model=prequant_model,
        data_seed=data_seed,
        output_paths=output_paths,
        step_nvtx_enabled=step_nvtx_enabled,
    )
    resume_trainer_state = (
        load_zo_trainer_state(args.resume_lora_checkpoint)
        if args.resume_lora_checkpoint
        else None
    )
    if resume_trainer_state is not None:
        print(
            "[vLLM] resume_trainer_state="
            f"{args.resume_lora_checkpoint} "
            f"global_step={resume_trainer_state.get('global_step')} "
            f"log_history={len(resume_trainer_state.get('log_history', []))}",
            flush=True,
        )
    wandb_logger = init_wandb_logger(
        args=args,
        resume_trainer_state=resume_trainer_state,
    )
    wandb_run = wandb_logger.run

    model_config = AutoConfig.from_pretrained(model_name)
    perturb_tied_lm_head = perturb_embeddings and bool(
        getattr(model_config, "tie_word_embeddings", False)
    )
    resolved_target_modules = resolve_lora_target_modules(
        None,
        include_lm_head=perturb_tied_lm_head,
        include_embeddings=perturb_embeddings,
    )
    tokenizer = build_tokenizer(
        model_name,
        opt_bos_mode=args.opt_bos_mode,
        use_fast=False,
    )
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": bool(int(args.enforce_eager)),
        "enable_lora": True,
        "max_lora_rank": lora_slot_rank,
        "max_loras": 2,
        "lora_target_modules": list(resolved_target_modules),
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.vllm_quantization:
        llm_kwargs["quantization"] = args.vllm_quantization
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
    objective_max_length = int(
        args.max_model_len
        or getattr(model_config, "max_position_embeddings", None)
        or 2048
    )
    weight_sync = WeightSync(
        llm,
        num_layers=model_config.num_hidden_layers,
        model_config=model_config,
    )

    config = VLLMZOConfig(
        estimator=args.estimator,
        direction_provider=args.direction_provider,
        lozo_provider_mode=args.lozo_provider_mode,
        rank=args.rank,
        eps=args.eps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        nu=args.nu,
        random_device=args.zo_random_device,
        direction_sampling=args.direction_sampling,
        direction_scale=effective_direction_scale,
        perturbation_normalization=args.perturbation_normalization,
        power_iter_steps=args.agzo_power_iter_steps,
        low_rank_oversample=args.agzo_low_rank_oversample,
        basis_method=args.agzo_basis_method,
        u_dim=args.u_dim,
        max_logits_tokens=args.direct_worker_max_logits_tokens,
        loss_impl=args.direct_worker_loss_impl,
        score_chunk_size=args.score_chunk_size,
        activation_force_eager=bool(int(args.agzo_activation_force_eager)),
        seed=args.seed,
        num_queries=args.num_queries,
        perturbation_sides=args.perturbation_sides,
        query_microbatch_size=args.query_microbatch_size,
    )
    param_metadata = weight_sync.get_hf_param_metadata(
        include_embeddings=perturb_embeddings,
    )
    zo_engine = ZOVLLMEngine(
        model=model_name,
        rank=args.rank,
        config=ZOVLLMEngineConfig(
            lora_rank=lora_slot_rank,
            target_modules=resolved_target_modules,
        ),
        llm=llm,
        model_config=model_config,
    )
    lora_runtimes = [zo_engine.runtime]
    for runtime in lora_runtimes:
        runtime.register_slots()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.train_objective == "prompt_nll":
        prompts = prepare_prompt_nll_sst2_data(data_seed, num_samples=args.num_samples)
        dev_rows = []
        valid_rows_cls = []
        train_items = tokenize_prompts(tokenizer, prompts)
    else:
        train_items, dev_rows, valid_rows_cls = load_objective_rows(
            args.train_objective,
            data_seed=data_seed,
            num_train=args.num_samples,
            num_dev=args.num_dev,
            num_eval=args.eval_accuracy_samples,
        )
    train_dataloader = make_train_dataloader(
        train_items,
        args.batch_size,
        sampler=args.train_sampler,
        seed=dataloader_seed,
        drop_last=bool(args.dataloader_drop_last),
    )
    if len(train_dataloader) <= 0:
        raise SystemExit(
            "no full training batches; increase --num-samples or lower --batch-size"
        )
    intervals = resolve_runner_intervals(
        args=args,
        num_train_items=len(train_items),
    )
    effective_eval_interval = intervals.eval
    effective_progress_interval = intervals.progress
    effective_train_loss_interval = intervals.train_loss
    effective_save_steps = intervals.save
    print_interval_summary(
        args=args,
        batches_per_epoch=len(train_dataloader),
        intervals=intervals,
    )
    subspace_row_stream = list(train_items)
    if not subspace_row_stream:
        raise SystemExit("no rows available for AGZO subspace construction")
    current_raw_step = {"value": 1}
    task_collator = ZOTaskDataCollator(
        tokenizer=tokenizer,
        config=ZOTaskEncodingConfig(
            objective_name=args.train_objective,
            max_length=objective_max_length,
            max_new_tokens=args.max_new_tokens,
            direction_provider=args.direction_provider,
            agzo_kappa=int(args.agzo_kappa),
        ),
        subspace_rows=subspace_row_stream,
        step_getter=lambda: int(current_raw_step["value"]),
    )
    if args.base_eval_mode == "skip":
        initial_loss = None
        print("[vLLM] initial_loss=skipped", flush=True)
    else:
        initial_loss = score_objective_loss(
            args.train_objective,
            llm,
            dev_rows,
            tokenizer,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
            max_length=objective_max_length,
            max_new_tokens=args.max_new_tokens,
        )
        if initial_loss is None:
            print("[vLLM] initial_loss=skipped", flush=True)
        else:
            print(f"[vLLM] initial_loss={initial_loss:.6f}", flush=True)

    np.random.seed(args.seed)
    eval_losses = (
        [] if initial_loss is None else [{"step": 0, "loss": float(initial_loss)}]
    )
    if accuracy_eval_mode == "skip":
        initial_dev_acc = None
        initial_valid_acc = None
        print("[vLLM] initial_acc=skipped", flush=True)
    else:
        initial_metrics = eval_objective_metrics(
            args.train_objective,
            llm,
            tokenizer,
            dev_rows,
            valid_rows_cls,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
            max_length=objective_max_length,
            max_new_tokens=args.max_new_tokens,
        )
        if initial_metrics is None:
            initial_dev_acc = None
            initial_valid_acc = None
            print("[vLLM] initial_acc=skipped", flush=True)
        else:
            initial_dev_acc = initial_metrics.dev_value
            initial_valid_acc = initial_metrics.valid_value
            if initial_metrics.primary_name == "f1":
                initial_dev_metric = initial_metrics.dev_metrics
                initial_valid_metric = initial_metrics.valid_metrics
                if initial_dev_metric is not None:
                    print(
                        f"[vLLM] initial_f1={initial_dev_metric['f1']:.6f} "
                        f"initial_em={initial_dev_metric['em']:.6f}",
                        flush=True,
                    )
                if initial_valid_metric is not None:
                    print(
                        f"[vLLM] initial_valid_f1={initial_valid_metric['f1']:.6f} "
                        f"initial_valid_em={initial_valid_metric['em']:.6f}",
                        flush=True,
                    )
                wandb_logger.log(
                    {
                        "eval/loss": None
                        if initial_loss is None
                        else float(initial_loss),
                        "eval/f1": initial_dev_acc,
                        "eval/em": None
                        if initial_dev_metric is None
                        else initial_dev_metric["em"],
                        "eval_valid/f1": initial_valid_acc,
                        "eval_valid/em": None
                        if initial_valid_metric is None
                        else initial_valid_metric["em"],
                    },
                    0,
                )
            else:
                if initial_dev_acc is not None:
                    print(f"[vLLM] initial_acc={initial_dev_acc:.6f}", flush=True)
                if initial_valid_acc is not None:
                    print(
                        f"[vLLM] initial_valid_acc={initial_valid_acc:.6f}",
                        flush=True,
                    )
                wandb_logger.log(
                    {
                        "eval/loss": None
                        if initial_loss is None
                        else float(initial_loss),
                        "eval/accuracy": initial_dev_acc,
                        "eval_valid/accuracy": initial_valid_acc,
                    },
                    0,
                )
    eval_metrics = (
        []
        if initial_loss is None
        else [
            {
                "step": 0,
                "loss": float(initial_loss),
                "accuracy": initial_dev_acc,
                "valid_accuracy": initial_valid_acc,
            }
        ]
    )
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

    checkpoint_settings = resolve_runtime_checkpoint_settings(
        args=args,
        effective_eval_interval=effective_eval_interval,
        effective_save_steps=effective_save_steps,
    )
    load_best_model_at_end = checkpoint_settings.load_best_model_at_end
    effective_save_steps = checkpoint_settings.effective_save_steps

    ckpt_root = os.path.join(output_dir, "checkpoints")
    runtime_checkpoints_enabled = checkpoint_settings.runtime_checkpoints_enabled
    if runtime_checkpoints_enabled:
        os.makedirs(ckpt_root, exist_ok=True)
    ckpt_paths = []
    checkpoint_records = []
    checkpoint_mode = checkpoint_settings.checkpoint_mode
    greater_is_better = (
        None if args.greater_is_better == "auto" else bool(int(args.greater_is_better))
    )
    best_tracker = BestMetricTracker(
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=greater_is_better,
    )
    restored_trainer_state_info = merge_zo_trainer_state(
        resume_trainer_state,
        history=history,
        eval_losses=eval_losses,
        eval_metrics=eval_metrics,
        checkpoint_records=checkpoint_records,
        checkpoint_paths=ckpt_paths,
        best_tracker=best_tracker,
    )
    if restored_trainer_state_info.get("restored"):
        print(
            "[vLLM] restored_log_continuity="
            f"history_rows={restored_trainer_state_info['history_rows']} "
            f"eval_metric_rows={restored_trainer_state_info['eval_metric_rows']} "
            f"checkpoint_records={restored_trainer_state_info['checkpoint_records']}",
            flush=True,
        )
    loaded_best_checkpoint = None
    if checkpoint_mode == "lora" and not use_lora_bank_update:
        raise ValueError(
            "save_checkpoint_mode='lora' currently requires "
            "--quantized-update-mode lora_bank"
        )
    if args.resume_lora_checkpoint:
        if not use_lora_bank_update:
            raise ValueError(
                "--resume-lora-checkpoint currently requires "
                "--quantized-update-mode lora_bank"
            )
    u_snapshot_root = os.path.join(output_dir, "u_snapshots")
    accumulated_update_state = None
    resume_lora_info = None
    if use_lora_bank_update:
        accumulated_update_state = BlockLoRAUpdateBankState(
            engine=zo_engine,
            update_bank_rank=int(args.update_bank_rank),
            u_beta=args.u_beta,
            u_norm_cap=args.u_norm_cap,
            gradient_accumulation_update_steps=int(
                args.gradient_accumulation_update_steps
            ),
        )
        if args.resume_lora_checkpoint:
            resume_device = "cuda" if torch.cuda.is_available() else "cpu"
            resume_lora_info = load_lora_bank_checkpoint(
                args.resume_lora_checkpoint,
                accumulated_update_state=accumulated_update_state,
                device=resume_device,
            )
            print(
                f"[vLLM] resume_lora_checkpoint={args.resume_lora_checkpoint} "
                f"resume_step={resume_lora_info['step']} "
                f"resume_raw_step={resume_lora_info['raw_step']} "
                f"num_bank_modules={resume_lora_info['num_bank_modules']} "
                f"max_used_rank={resume_lora_info['max_used_rank']}",
                flush=True,
            )
    elif use_accumulated_update:
        accumulated_update_state = AccumulatedLowRankUpdateState(
            weight_sync=weight_sync,
            engine=zo_engine,
            precision=args.weight_update_precision,
            sync_device=bool(int(args.sync_weight_update)),
            qkv_update_mode=args.qkv_weight_update,
            u_beta=args.u_beta,
            u_norm_cap=args.u_norm_cap,
            gradient_accumulation_update_steps=int(
                args.gradient_accumulation_update_steps
            ),
        )
    zo_model = VLLMZOModel(
        engine=zo_engine,
        weight_sync=weight_sync,
        config=config,
        param_metadata=param_metadata,
        update_state=accumulated_update_state,
        weight_update_precision=args.weight_update_precision,
        sync_weight_update=bool(int(args.sync_weight_update)),
        qkv_update_mode=args.qkv_weight_update,
    )
    if args.resume_lora_checkpoint:
        restore_info = restore_direction_provider_v_cache_from_lora_bank(
            param_metadata=param_metadata,
            accumulated_update_state=accumulated_update_state,
            direction_provider=zo_model.direction_provider,
            direction_provider_name=args.direction_provider,
            direction_scale=effective_direction_scale,
            v_normalization=config.v_normalization,
        )
        if hasattr(zo_model.direction_provider, "step"):
            zo_model.direction_provider.step = int(resume_lora_info["raw_step"])
        print(
            f"[vLLM] resume_lora_v_cache_restored="
            f"{restore_info['restored_v_cache_modules']} "
            f"subspace_queue_restored={restore_info['restored_subspace_queue']}",
            flush=True,
        )

    def update_accumulated_lora_for_clean_score(runtime, step_value):
        if not use_accumulated_lora_eval or accumulated_update_state is None:
            return None, 0.0
        return accumulated_update_state.set_clean_lora_for_score(
            step=step_value,
            runtime=runtime,
        )

    def flush_pending_updates_for_clean_score(step_value):
        if accumulated_update_state is None:
            return 0.0
        flush = getattr(accumulated_update_state, "flush_pending_to_accumulated", None)
        if not callable(flush):
            return 0.0
        return float(flush(step=step_value))

    u_snapshot_recorder = USnapshotRecorder(
        root=u_snapshot_root,
        interval=args.u_snapshot_interval,
        dtype_name=args.u_snapshot_dtype,
        total_limit=args.u_snapshot_total_limit,
        update_state=accumulated_update_state,
        log_wandb=wandb_logger.log,
    )

    checkpoint_manager = RuntimeCheckpointManager(
        args=args,
        checkpoint_mode=checkpoint_mode,
        checkpoint_root=ckpt_root,
        effective_save_steps=effective_save_steps,
        llm=llm,
        accumulated_update_state=accumulated_update_state,
        weight_sync=weight_sync,
        use_lora_bank_update=use_lora_bank_update,
        best_tracker=best_tracker,
        timing=timing,
        history=history,
        eval_losses=eval_losses,
        eval_metrics=eval_metrics,
        checkpoint_records=checkpoint_records,
        checkpoint_paths=ckpt_paths,
        wandb_run=wandb_run,
    )

    resume_measured_step = (
        0 if resume_lora_info is None else int(resume_lora_info["step"])
    )
    if resume_measured_step < 0:
        raise RuntimeError(f"invalid resume step: {resume_measured_step}")
    if resume_measured_step >= int(args.steps):
        raise RuntimeError(
            "--resume-lora-checkpoint step is already at or beyond --steps: "
            f"{resume_measured_step} >= {args.steps}"
        )
    resume_raw_step = (
        resume_measured_step + int(args.warmup_steps)
        if resume_lora_info is None
        else int(resume_lora_info["raw_step"])
    )
    measured_train_t0 = None
    measured_train_t1 = None
    last_progress_time = None
    last_progress_step = resume_measured_step
    current_eval_row = None
    last_raw_step = resume_raw_step
    last_step_context = {}

    class RunnerStepModel:
        def step(self, batch_rows, *, step: int):
            nonlocal measured_train_t0
            nonlocal measured_train_t1
            nonlocal last_raw_step
            last_raw_step = int(step)
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
            lora_runtime = lora_runtimes[0]
            record_timing("prep_wait_s", 0.0)
            record_timing("prep_launch_s", 0.0)
            direction_digest = None
            if args.direction_digest:
                print(
                    "[vLLM] direction_digest=skipped_unified_stepper",
                    flush=True,
                )
            if args.trace_step_events:
                print(f"[trace] step={step} unified_step_start", flush=True)
            stepper_t0 = time.perf_counter()
            with nvtx_range("zo_step.unified", emit_step_nvtx):
                current_raw_step["value"] = int(step)
                collate_t0 = time.perf_counter()
                zo_batch = task_collator(batch_rows)
                runner_collate_s = time.perf_counter() - collate_t0
                model_step_t0 = time.perf_counter()
                stepper_result = zo_model.step(
                    zo_batch,
                    step=step,
                )
                runner_model_step_s = time.perf_counter() - model_step_t0
            unpack_t0 = time.perf_counter()
            stepper_profile = dict(stepper_result.profile_s)
            stepper_update = dict(
                getattr(stepper_result, "update_info", None)
                or getattr(stepper_result, "weight_update_info", {})
            )
            direction_info = dict(
                getattr(stepper_result, "direction_info", None)
                or getattr(stepper_result, "subspace_info", {})
            )
            random_seed = direction_info.get("perturb_seed")
            loss_plus = float(stepper_result.loss_plus)
            loss_minus = float(stepper_result.loss_minus)
            c = float(stepper_result.projected_grad)
            runner_unpack_s = time.perf_counter() - unpack_t0
            record_timing("direction_s", 0.0)
            record_timing("build_lora_s", 0.0)
            record_timing(
                "lora_update_s",
                float(stepper_profile.get("set_plus_minus_directions", 0.0)),
            )
            record_timing(
                "score_s",
                float(
                    stepper_profile.get(
                        "score_plus_minus_total",
                        time.perf_counter() - stepper_t0,
                    )
                ),
            )
            record_timing(
                "weight_update_s",
                float(
                    stepper_update.get("accumulate_s", 0.0)
                    + stepper_update.get("fold_s", 0.0)
                ),
            )
            record_timing("weight_fold_s", float(stepper_update.get("fold_s", 0.0)))
            if args.profile_mode == "detailed" and measured_step:
                if "score_token_groups" in stepper_profile:
                    timing["score_direct_worker_s"].append(
                        float(stepper_profile["score_token_groups"])
                    )
                timing.setdefault("runner_collate_s", []).append(
                    float(runner_collate_s)
                )
                timing.setdefault("runner_model_step_s", []).append(
                    float(runner_model_step_s)
                )
                timing.setdefault("runner_unpack_s", []).append(float(runner_unpack_s))
                for key, value in stepper_profile.items():
                    timing.setdefault(f"score_worker_unified_{key}_s", []).append(
                        float(value)
                    )
            if args.trace_step_events:
                print(f"[trace] step={step} unified_step_done", flush=True)
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
                last_step_context.clear()
                last_step_context.update(
                    {
                        "batch_rows": batch_rows,
                        "raw_step": int(step),
                        "measured_step": True,
                        "measured_index": int(measured_index),
                        "lora_runtime": lora_runtime,
                        "seed": int(random_seed),
                        "loss_plus": float(loss_plus),
                        "loss_minus": float(loss_minus),
                        "projected_grad": float(c),
                        "direction_digest": direction_digest,
                        "step_s": float(timing["step_s"][-1]),
                    }
                )
                return {
                    "loss_plus": float(loss_plus),
                    "loss_minus": float(loss_minus),
                    "projected_grad": float(c),
                    "step_s": float(timing["step_s"][-1]) if measured_step else 0.0,
                }
            return {
                "loss_plus": float(loss_plus),
                "loss_minus": float(loss_minus),
                "projected_grad": float(c),
                "step_s": 0.0,
            }

    class VLLMTaskRunnerCallback(VLLMZOTrainerCallback):
        def on_step_end(self, callback_args, state, control, **kwargs):
            nonlocal last_progress_time
            nonlocal last_progress_step
            nonlocal current_eval_row
            if not last_step_context.get("measured_step"):
                return None
            measured_index = int(last_step_context["measured_index"])
            raw_step = int(last_step_context["raw_step"])
            lora_runtime = last_step_context["lora_runtime"]
            loss_plus = float(last_step_context["loss_plus"])
            loss_minus = float(last_step_context["loss_minus"])
            c = float(last_step_context["projected_grad"])
            random_seed = int(last_step_context["seed"])
            batch_rows = last_step_context["batch_rows"]
            current_eval_row = None

            if args.record_history:
                history.append(
                    {
                        "step": measured_index,
                        "raw_step": raw_step,
                        "seed": random_seed,
                        "loss_plus": loss_plus,
                        "loss_minus": loss_minus,
                        "c": c,
                        "direction_digest": last_step_context["direction_digest"],
                        "step_s": float(last_step_context["step_s"]),
                    }
                )

            if (
                effective_train_loss_interval > 0
                and measured_index % effective_train_loss_interval == 0
            ):
                train_loss_lora_id = None
                train_loss_lora_s = 0.0
                if use_accumulated_lora_eval:
                    train_loss_lora_id, train_loss_lora_s = (
                        update_accumulated_lora_for_clean_score(
                            lora_runtime,
                            raw_step,
                        )
                    )
                train_loss_fold_s = 0.0
                train_loss, train_loss_score_s = score_current_batch_train_loss(
                    args.train_objective,
                    llm,
                    batch_rows,
                    tokenizer,
                    max_logits_tokens=args.direct_worker_max_logits_tokens,
                    loss_impl=args.direct_worker_loss_impl,
                    max_length=objective_max_length,
                    max_new_tokens=args.max_new_tokens,
                    lora_id=train_loss_lora_id,
                )
                print(
                    f"[vLLM] step={measured_index} "
                    f"train_loss={train_loss:.6f} "
                    f"train_loss_scope=current_batch_unperturbed "
                    f"train_loss_score_s={train_loss_score_s:.6f} "
                    f"train_loss_weight_fold_s={train_loss_fold_s:.6f} "
                    f"train_loss_accum_lora_s={train_loss_lora_s:.6f} "
                    f"train_loss_lora_id={train_loss_lora_id} "
                    f"learning_rate={args.lr:.6g}",
                    flush=True,
                )
                wandb_logger.log(
                    {
                        "train/loss": float(train_loss),
                        "train/loss_score_s": float(train_loss_score_s),
                        "train/weight_fold_s": float(train_loss_fold_s),
                        "train/accum_lora_s": float(train_loss_lora_s),
                        "train/learning_rate": float(args.lr),
                    },
                    measured_index,
                )

            if (
                effective_progress_interval > 0
                and measured_index % effective_progress_interval == 0
            ):
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
                    f"[vLLM] step={measured_index} seed={random_seed} "
                    f"plus={loss_plus:.6f} minus={loss_minus:.6f} "
                    f"c={c:.6f} step_s={last_step_context['step_s']:.6f} "
                    f"window_steps={window_steps} window_s={window_s:.6f} "
                    f"cumulative_s={cumulative_s:.6f}",
                    flush=True,
                )
                wandb_logger.log(
                    {
                        "probe/loss_plus": loss_plus,
                        "probe/loss_minus": loss_minus,
                        "probe/c": c,
                        "perf/step_s": float(last_step_context["step_s"]),
                        "perf/window_s": float(window_s),
                        "perf/cumulative_s": float(cumulative_s),
                    },
                    measured_index,
                )
                last_progress_time = progress_now
                last_progress_step = measured_index

            return None

        def run_periodic_eval_from_context(self):
            nonlocal current_eval_row
            if not last_step_context.get("measured_step"):
                return {"loss": None}
            current_eval_row = self._run_periodic_eval(
                measured_index=int(last_step_context["measured_index"]),
                raw_step=int(last_step_context["raw_step"]),
                lora_runtime=last_step_context["lora_runtime"],
            )
            return {
                "eval_loss": current_eval_row.get("loss"),
                "eval_accuracy": current_eval_row.get("accuracy"),
                "eval_valid_accuracy": current_eval_row.get("valid_accuracy"),
            }

        def _run_periodic_eval(self, *, measured_index, raw_step, lora_runtime):
            if args.base_eval_mode == "skip":
                print(
                    f"[vLLM] step={measured_index} eval_loss=skipped",
                    flush=True,
                )
                u_snapshot_recorder.capture(measured_index, None, None)
                return {"step": measured_index, "loss": None}
            eval_lora_id = None
            eval_lora_s = 0.0
            pending_flush_s = 0.0
            if use_accumulated_lora_eval:
                pending_flush_s = flush_pending_updates_for_clean_score(raw_step)
                if pending_flush_s:
                    print(
                        f"[vLLM] step={measured_index} "
                        f"eval_pending_flush_s={pending_flush_s:.6f}",
                        flush=True,
                    )
                eval_lora_id, eval_lora_s = update_accumulated_lora_for_clean_score(
                    lora_runtime,
                    raw_step,
                )
            if eval_lora_id is not None:
                print(
                    f"[vLLM] step={measured_index} "
                    f"eval_accum_lora_id={eval_lora_id} "
                    f"eval_accum_lora_s={eval_lora_s:.6f}",
                    flush=True,
                )
            periodic_eval = score_periodic_eval(
                args.train_objective,
                llm,
                tokenizer,
                dev_rows,
                valid_rows_cls,
                accuracy_eval_mode=accuracy_eval_mode,
                max_logits_tokens=args.direct_worker_max_logits_tokens,
                loss_impl=args.direct_worker_loss_impl,
                max_length=objective_max_length,
                max_new_tokens=args.max_new_tokens,
                lora_id=eval_lora_id,
            )
            if periodic_eval is None:
                print(
                    f"[vLLM] step={measured_index} eval_loss=skipped",
                    flush=True,
                )
                return {"step": measured_index, "loss": None}
            val_loss = periodic_eval.loss
            val_acc = periodic_eval.dev_value
            valid_acc = periodic_eval.valid_value
            eval_losses.append({"step": measured_index, "loss": float(val_loss)})
            eval_row = {
                "step": measured_index,
                "loss": float(val_loss),
                "accuracy": val_acc,
                "valid_accuracy": valid_acc,
            }
            eval_metrics.append(eval_row)
            print(
                f"[vLLM] step={measured_index} eval_loss={val_loss:.6f}",
                flush=True,
            )
            if periodic_eval.primary_name == "f1":
                val_metric = periodic_eval.dev_metrics
                if val_metric is not None:
                    print(
                        f"[vLLM] step={measured_index} "
                        f"eval_f1={val_metric['f1']:.6f} "
                        f"eval_em={val_metric['em']:.6f}",
                        flush=True,
                    )
                wandb_payload = {
                    "eval/loss": float(val_loss),
                    "eval/f1": val_acc,
                    "eval/em": None if val_metric is None else val_metric["em"],
                    "eval/accum_lora_s": float(eval_lora_s),
                    "eval/pending_flush_s": float(pending_flush_s),
                }
            else:
                if val_acc is not None:
                    print(
                        f"[vLLM] step={measured_index} eval_acc={val_acc:.6f}",
                        flush=True,
                    )
                if valid_acc is not None:
                    print(
                        f"[vLLM] step={measured_index} eval_valid_acc={valid_acc:.6f}",
                        flush=True,
                    )
                wandb_payload = {
                    "eval/loss": float(val_loss),
                    "eval/accuracy": val_acc,
                    "eval_valid/accuracy": valid_acc,
                    "eval/accum_lora_s": float(eval_lora_s),
                    "eval/pending_flush_s": float(pending_flush_s),
                }
            u_snapshot_recorder.capture(measured_index, val_loss, val_acc)
            wandb_logger.log(wandb_payload, measured_index)
            if periodic_eval.primary_name == "f1" and valid_acc is not None:
                print(
                    f"[vLLM] step={measured_index} eval_valid_f1={valid_acc:.6f}",
                    flush=True,
                )
            return eval_row

    task_callback = VLLMTaskRunnerCallback()

    def save_runtime_checkpoint_from_trainer(checkpoint_dir):
        if not last_step_context.get("measured_step"):
            return None
        measured_index = int(last_step_context["measured_index"])
        raw_step = int(last_step_context["raw_step"])
        reason = "best" if args.save_strategy == "best" else "steps"
        checkpoint_eval_row = None
        if current_eval_row is not None and current_eval_row["loss"] is not None:
            checkpoint_eval_row = {
                "step": int(current_eval_row["step"]),
                "eval_loss": current_eval_row["loss"],
                "eval_accuracy": current_eval_row["accuracy"],
                "eval_valid_accuracy": current_eval_row["valid_accuracy"],
            }
        return checkpoint_manager.save_runtime_checkpoint(
            measured_index,
            raw_step=raw_step,
            eval_row=checkpoint_eval_row,
            reason=reason,
            checkpoint_path=checkpoint_dir,
        )

    trainer_args = ZOTrainingArguments(
        output_dir=output_dir,
        max_steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
        per_device_train_batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        logging_steps=max(1, int(effective_progress_interval or args.steps or 1)),
        eval_steps=max(1, int(effective_eval_interval or args.steps or 1)),
        save_steps=max(0, int(effective_save_steps)),
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=best_tracker.greater_is_better,
        seed=int(args.seed),
        dataloader_drop_last=bool(args.dataloader_drop_last),
    )
    trainer = VLLMZOTrainer(
        model=RunnerStepModel(),
        args=trainer_args,
        train_dataloader=train_dataloader,
        eval_fn=(
            task_callback.run_periodic_eval_from_context
            if effective_eval_interval > 0
            else None
        ),
        save_model_fn=(
            save_runtime_checkpoint_from_trainer
            if runtime_checkpoints_enabled
            else None
        ),
        callbacks=[task_callback],
    )
    trainer.set_initial_state(
        global_step=int(resume_measured_step),
        raw_step=int(resume_raw_step),
    )
    trainer.train()
    total_s = (
        0.0 if measured_train_t0 is None else measured_train_t1 - measured_train_t0
    )
    if load_best_model_at_end:
        if best_tracker.best_record is None:
            raise RuntimeError(
                "load_best_model_at_end was enabled, but no best checkpoint was "
                "recorded. Check save_steps/eval_interval and metric settings."
            )
        best_checkpoint = best_tracker.best_record.get("checkpoint")
        if best_checkpoint is None or not best_checkpoint.get("loadable"):
            raise RuntimeError(
                "load_best_model_at_end requires a loadable native checkpoint; "
                f"best_record={best_tracker.best_record}"
            )
        best_checkpoint_path = str(best_checkpoint["path"])
        load_results = load_native_checkpoint_into_workers(llm, best_checkpoint_path)
        clear_update_state_for_loaded_checkpoint(accumulated_update_state)
        loaded_best_checkpoint = dict(best_tracker.best_record) | {
            "loaded_checkpoint_path": best_checkpoint_path,
            "load_results": load_results,
            "loaded_at_step": int(args.steps),
        }
        print(
            f"[vLLM] loaded_best_checkpoint={best_checkpoint_path} "
            f"metric={loaded_best_checkpoint['metric']} "
            f"value={loaded_best_checkpoint['metric_value']:.6f}",
            flush=True,
        )

    final_lora_id = None
    final_lora_s = 0.0
    if use_accumulated_lora_eval:
        final_pending_flush_s = flush_pending_updates_for_clean_score(last_raw_step)
        if final_pending_flush_s:
            print(
                f"[vLLM] final_pending_flush_s={final_pending_flush_s:.6f}",
                flush=True,
            )
        final_lora_id, final_lora_s = update_accumulated_lora_for_clean_score(
            lora_runtimes[0],
            last_raw_step,
        )
    if final_lora_id is not None:
        print(
            f"[vLLM] final_accum_lora_id={final_lora_id} "
            f"final_accum_lora_s={final_lora_s:.6f}",
            flush=True,
        )
    if args.base_eval_mode == "skip":
        final_loss = None
        print("[vLLM] final_loss=skipped", flush=True)
    else:
        final_loss = score_objective_loss(
            args.train_objective,
            llm,
            dev_rows,
            tokenizer,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
            max_length=objective_max_length,
            max_new_tokens=args.max_new_tokens,
            lora_id=final_lora_id,
        )
        if final_loss is None:
            print("[vLLM] final_loss=skipped", flush=True)
        else:
            if not eval_losses or eval_losses[-1]["step"] != args.steps:
                eval_losses.append({"step": args.steps, "loss": float(final_loss)})
            print(f"[vLLM] final_loss={final_loss:.6f}", flush=True)
    final_dev_acc = None
    final_valid_acc = None
    if accuracy_eval_mode == "skip":
        print("[vLLM] final_acc=skipped", flush=True)
    else:
        final_metrics = eval_objective_metrics(
            args.train_objective,
            llm,
            tokenizer,
            dev_rows,
            valid_rows_cls,
            max_logits_tokens=args.direct_worker_max_logits_tokens,
            loss_impl=args.direct_worker_loss_impl,
            max_length=objective_max_length,
            max_new_tokens=args.max_new_tokens,
            lora_id=final_lora_id,
        )
        if final_metrics is None:
            print("[vLLM] final_acc=skipped", flush=True)
        else:
            final_dev_acc = final_metrics.dev_value
            final_valid_acc = final_metrics.valid_value
            if final_metrics.primary_name == "f1":
                final_dev_metric = final_metrics.dev_metrics
                final_valid_metric = final_metrics.valid_metrics
                if final_dev_metric is not None:
                    print(
                        f"[vLLM] final_f1={final_dev_metric['f1']:.6f} "
                        f"final_em={final_dev_metric['em']:.6f}",
                        flush=True,
                    )
                if final_valid_metric is not None:
                    print(
                        f"[vLLM] final_valid_f1={final_valid_metric['f1']:.6f} "
                        f"final_valid_em={final_valid_metric['em']:.6f}",
                        flush=True,
                    )
    if not eval_metrics or eval_metrics[-1]["step"] != args.steps:
        eval_metrics.append(
            {
                "step": args.steps,
                "loss": None if final_loss is None else float(final_loss),
                "accuracy": final_dev_acc,
                "valid_accuracy": final_valid_acc,
            }
        )
    if final_dev_acc is not None and args.train_objective == "squad_nll":
        print(f"[vLLM] final_dev_f1={final_dev_acc:.6f}", flush=True)
    elif final_dev_acc is not None:
        print(f"[vLLM] final_dev_acc={final_dev_acc:.6f}", flush=True)
    if final_valid_acc is not None and args.train_objective == "squad_nll":
        print(f"[vLLM] final_valid_f1={final_valid_acc:.6f}", flush=True)
    elif final_valid_acc is not None:
        print(f"[vLLM] final_acc={final_valid_acc:.6f}", flush=True)
    u_snapshot_recorder.capture(args.steps, final_loss, final_dev_acc)
    final_wandb_payload = {
        "final/loss": None if final_loss is None else float(final_loss),
        "perf/total_s": float(total_s),
    }
    if args.train_objective == "squad_nll":
        final_wandb_payload |= {
            "final/dev_f1": final_dev_acc,
            "final/valid_f1": final_valid_acc,
        }
    else:
        final_wandb_payload |= {
            "final/dev_accuracy": final_dev_acc,
            "final/valid_accuracy": final_valid_acc,
        }
    wandb_logger.log(final_wandb_payload, args.steps)
    checkpoint_manager.save_final_native_checkpoint(args.steps, raw_step=last_raw_step)

    output_file = write_vllm_perf_json(
        output_dir=output_dir,
        profile_mode=args.profile_mode,
        timestamp=timestamp,
        args=args,
        model_name=model_name,
        effective_direction_scale=effective_direction_scale,
        direction_scale_mode=direction_scale_mode,
        direction_scale_note=direction_scale_note,
        direction_scale_applies_to=direction_scale_applies_to,
        output_root_resolved=output_root_resolved,
        experiment_name_resolved=experiment_name_resolved,
        lora_slot_rank=lora_slot_rank,
        prequant_model=prequant_model,
        accuracy_eval_mode=accuracy_eval_mode,
        train_objective=args.train_objective,
        use_unified_stepper=use_unified_stepper,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_losses=eval_losses,
        eval_metrics=eval_metrics,
        initial_dev_acc=initial_dev_acc,
        initial_valid_acc=initial_valid_acc,
        final_dev_acc=final_dev_acc,
        final_valid_acc=final_valid_acc,
        history=history,
        timing=timing,
        total_s=total_s,
        ckpt_paths=ckpt_paths,
        checkpoint_records=checkpoint_records,
        best_checkpoint=best_tracker.best_record,
        loaded_best_checkpoint=loaded_best_checkpoint,
        u_snapshot_paths=u_snapshot_recorder.paths,
        u_snapshot_metrics=u_snapshot_recorder.metrics,
    )

    for runtime in lora_runtimes:
        runtime.cleanup()
    wandb_logger.finish()
    print(f"[vLLM] saved={output_file}", flush=True)


if __name__ == "__main__":
    main()
