import pytest

from zo_vllm.config import (
    DEFAULT_ZO_BATCH_SIZE,
    DEFAULT_ZO_DIRECTION_SCALE,
    DEFAULT_ZO_EPS,
    DEFAULT_ZO_GPU_MEMORY_UTILIZATION,
    DEFAULT_ZO_LEARNING_RATE,
    DEFAULT_ZO_LOZO_PROVIDER_MODE,
    DEFAULT_ZO_NU,
    DEFAULT_ZO_PERTURBATION_NORMALIZATION,
    DEFAULT_ZO_PERTURB_EMBEDDINGS,
    DEFAULT_ZO_RANK,
    DEFAULT_ZO_SEED,
    DEFAULT_ZO_STEPS,
    DEFAULT_ZO_TASK_SHUFFLE_IMPL,
    DEFAULT_ZO_WEIGHT_DECAY,
    VLLMZOConfig,
)
from zo_vllm.core.lora_scope import lora_scope_includes_embeddings
from zo_vllm.experiment.runners.args import build_vllm_zo_task_arg_parser
from zo_vllm.tasks.tokenization import OPT_BOS_NATIVE
from zo_vllm.training.arguments import ZOTrainingArguments
from zo_vllm.training.direction import (
    LOZOFastDirectionProvider,
)


def test_task_runner_defaults_follow_package_config():
    args = build_vllm_zo_task_arg_parser().parse_args([])

    assert DEFAULT_ZO_RANK == 2
    assert DEFAULT_ZO_LEARNING_RATE == 1e-7
    assert DEFAULT_ZO_GPU_MEMORY_UTILIZATION == 0.9
    assert DEFAULT_ZO_LOZO_PROVIDER_MODE == "fast"
    assert args.rank == DEFAULT_ZO_RANK
    assert args.nu == DEFAULT_ZO_NU
    assert args.lr == DEFAULT_ZO_LEARNING_RATE
    assert args.weight_decay == DEFAULT_ZO_WEIGHT_DECAY
    assert args.seed == DEFAULT_ZO_SEED
    assert args.direction_scale == DEFAULT_ZO_DIRECTION_SCALE
    assert args.perturbation_normalization == DEFAULT_ZO_PERTURBATION_NORMALIZATION
    assert args.perturb_embeddings == DEFAULT_ZO_PERTURB_EMBEDDINGS
    assert args.lozo_provider_mode == DEFAULT_ZO_LOZO_PROVIDER_MODE
    assert args.gpu_memory_utilization == DEFAULT_ZO_GPU_MEMORY_UTILIZATION
    assert args.dataloader_drop_last is False
    assert args.task_shuffle_impl == DEFAULT_ZO_TASK_SHUFFLE_IMPL
    assert args.eval_interval_epochs == 0.0
    assert args.progress_interval_epochs == 0.0
    assert args.train_loss_interval_epochs == 0.0
    assert args.opt_bos_mode == OPT_BOS_NATIVE
    assert args.u_snapshot_interval == 0


def test_task_runner_accepts_lora_checkpoint_mode():
    args = build_vllm_zo_task_arg_parser().parse_args(
        ["--save-checkpoint-mode", "lora"]
    )

    assert args.save_checkpoint_mode == "lora"


def test_task_runner_accepts_lora_full_scope():
    args = build_vllm_zo_task_arg_parser().parse_args(["--train-scope", "lora_full"])

    assert args.train_scope == "lora_full"
    assert lora_scope_includes_embeddings(args.train_scope)
    assert lora_scope_includes_embeddings("lora_normal", perturb_embeddings=True)
    assert not lora_scope_includes_embeddings("lora_normal")


def test_training_argument_defaults_follow_package_config(tmp_path):
    args = ZOTrainingArguments(output_dir=str(tmp_path))

    assert args.max_steps == DEFAULT_ZO_STEPS
    assert args.per_device_train_batch_size == DEFAULT_ZO_BATCH_SIZE
    assert args.per_device_eval_batch_size == DEFAULT_ZO_BATCH_SIZE
    assert args.learning_rate == DEFAULT_ZO_LEARNING_RATE
    assert args.seed == DEFAULT_ZO_SEED


def test_vllm_zo_config_defaults_are_main_path_defaults():
    config = VLLMZOConfig()

    assert config.estimator == "single_direction_antithetic"
    assert config.num_queries == 1
    assert config.perturbation_sides == "two_sided"
    assert config.query_microbatch_size == 2
    assert config.population_size == 30
    assert config.sigma == DEFAULT_ZO_EPS
    assert config.reward_shaping == "z_score"
    assert config.rank == DEFAULT_ZO_RANK
    assert config.nu == DEFAULT_ZO_NU
    assert config.direction_scale == DEFAULT_ZO_DIRECTION_SCALE
    assert config.perturbation_normalization == DEFAULT_ZO_PERTURBATION_NORMALIZATION
    assert config.lozo_provider_mode == DEFAULT_ZO_LOZO_PROVIDER_MODE
    assert config.seed == DEFAULT_ZO_SEED


def test_vllm_zo_config_accepts_infinite_nu():
    config = VLLMZOConfig(nu=-1)

    assert config.nu == -1


@pytest.mark.parametrize("field", ["learning_rate", "weight_decay"])
def test_vllm_zo_config_rejects_optimizer_policy(field):
    with pytest.raises(TypeError):
        VLLMZOConfig(**{field: 0.0})


def test_vllm_zo_config_keeps_lazy_v_lozo_path():
    config = VLLMZOConfig(
        estimator="single_direction_antithetic",
        direction_provider="lozo",
        lozo_provider_mode="fast",
        nu=50,
    )

    assert config.estimator == "single_direction_antithetic"
    assert config.direction_provider == "lozo"
    assert config.lozo_provider_mode == "fast"
    assert config.nu == 50


def test_vllm_zo_config_accepts_high_rank_mezo_like_path():
    config = VLLMZOConfig(
        estimator="single_direction_antithetic",
        direction_provider="lozo",
        lozo_provider_mode="scheduled",
        rank=512,
        nu=1,
    )

    assert config.direction_provider == "lozo"
    assert config.lozo_provider_mode == "scheduled"
    assert config.rank == 512
    assert config.nu == 1


def test_vllm_zo_config_accepts_multi_query_one_sided_path():
    config = VLLMZOConfig(
        estimator="multi_query",
        num_queries=4,
        perturbation_sides="one-sided",
        query_microbatch_size=2,
        multi_query_direction_mode="independent",
    )

    assert config.estimator == "multi_query"
    assert config.num_queries == 4
    assert config.perturbation_sides == "one_sided"
    assert config.query_microbatch_size == 2
    assert config.multi_query_direction_mode == "independent"


def test_vllm_zo_config_accepts_evolution_strategy_path():
    config = VLLMZOConfig(
        estimator="evolution_strategy",
        population_size=4,
        sigma=0.01,
        reward_shaping="z-scores",
        query_microbatch_size=2,
        multi_query_direction_mode="independent",
    )

    assert config.estimator == "evolution_strategy"
    assert config.population_size == 4
    assert config.sigma == 0.01
    assert config.reward_shaping == "z_score"
    assert config.multi_query_direction_mode == "independent"


def test_vllm_zo_config_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="unsupported estimator"):
        VLLMZOConfig(estimator="es")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_queries", 0, "num_queries"),
        ("perturbation_sides", "left", "perturbation_sides"),
        ("query_microbatch_size", 0, "query_microbatch_size"),
        ("multi_query_direction_mode", "dense", "multi_query_direction_mode"),
        ("population_size", 0, "population_size"),
        ("sigma", 0.0, "sigma"),
        ("reward_shaping", "rank", "reward_shaping"),
    ],
)
def test_vllm_zo_config_rejects_invalid_multi_query_params(field, value, message):
    with pytest.raises(ValueError, match=message):
        VLLMZOConfig(**{field: value})


def test_fast_lozo_provider_is_public_direction_provider():
    assert LOZOFastDirectionProvider.__name__ == "LOZOFastDirectionProvider"
