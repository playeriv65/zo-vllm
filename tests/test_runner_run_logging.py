from types import SimpleNamespace

from zo_vllm.experiment.runners.run_logging import (
    print_interval_summary,
    print_vllm_run_header,
)


def test_print_vllm_run_header_includes_key_runtime_fields(capsys, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    args = SimpleNamespace(
        steps=10,
        warmup_steps=1,
        batch_size=2,
        rank=4,
        lr=1e-7,
        eps=1e-3,
        estimator="multi_query",
        num_queries=2,
        perturbation_sides="one_sided",
        query_microbatch_size=2,
        direction_scale=1.0,
        perturbation_normalization="rms",
        profile_mode="minimal",
        enforce_eager="0",
        scoring_backend="direct_worker",
        sync_weight_update="0",
        qkv_weight_update="batched",
        direct_update_mode="accumulate",
        quantized_update_mode="none",
        update_bank_rank="auto",
        update_bank_rank_auto=16,
        update_bank_rank_requested="auto",
        vllm_quantization=None,
        gradient_accumulation_update_steps=0,
        u_beta=1.0,
        u_norm_cap=None,
        u_snapshot_interval=0,
        direction_provider="lozo",
        lozo_provider_mode="fast",
        nu=50,
        agzo_kappa=8,
        agzo_basis_method="qr",
        u_dim=None,
        direct_worker_loss_impl="logprobs",
        score_chunk_size=256,
        direct_lora_from_directions="1",
        direction_sampling="exact",
        train_objective="sst2_classification",
        opt_bos_mode="hf",
        seed=42,
    )
    runtime_modes = SimpleNamespace(
        effective_direction_scale=1.0,
        direction_scale_mode="explicit",
        direction_scale_note="user",
        direction_scale_applies_to="all",
        lora_slot_rank=16,
        use_unified_stepper=True,
    )
    output_paths = SimpleNamespace(
        output_dir="/tmp/run",
        output_root="/tmp",
        experiment_name="exp",
    )

    print_vllm_run_header(
        args=args,
        runtime_modes=runtime_modes,
        model_name="facebook/opt-125m",
        prequant_model=None,
        data_seed=43,
        output_paths=output_paths,
        step_nvtx_enabled=False,
    )

    out = capsys.readouterr().out
    assert "steps=10" in out
    assert "estimator=multi_query" in out
    assert "num_queries=2" in out
    assert "perturbation_sides=one_sided" in out
    assert "direction_provider=lozo" in out
    assert "lora_slot_rank=16" in out
    assert "output_dir=/tmp/run" in out


def test_print_interval_summary_includes_effective_intervals(capsys):
    args = SimpleNamespace(dataloader_drop_last=1, save_strategy="steps")
    intervals = SimpleNamespace(eval=100, progress=10, train_loss=25, save=50)

    print_interval_summary(
        args=args,
        batches_per_epoch=7,
        intervals=intervals,
    )

    out = capsys.readouterr().out
    assert "dataloader_drop_last=1" in out
    assert "batches_per_epoch=7" in out
    assert "eval_interval_steps=100" in out
    assert "save_steps=50" in out
