"""Argument parser definitions for experiment runners."""

from __future__ import annotations

import argparse

from zo_vllm.config import (
    DEFAULT_ZO_ACCURACY_EVAL_MODE,
    DEFAULT_ZO_BASE_EVAL_MODE,
    DEFAULT_ZO_BATCH_SIZE,
    DEFAULT_ZO_DATA_SEED,
    DEFAULT_ZO_DIRECT_LORA_FROM_DIRECTIONS,
    DEFAULT_ZO_DIRECT_UPDATE_MODE,
    DEFAULT_ZO_DIRECT_WORKER_LOSS_IMPL,
    DEFAULT_ZO_DIRECT_WORKER_MAX_LOGITS_TOKENS,
    DEFAULT_ZO_DIRECTION_PROVIDER,
    DEFAULT_ZO_DIRECTION_SAMPLING,
    DEFAULT_ZO_DIRECTION_SCALE,
    DEFAULT_ZO_ENFORCE_EAGER,
    DEFAULT_ZO_EPS,
    DEFAULT_ZO_EVAL_ACCURACY_SAMPLES,
    DEFAULT_ZO_EVAL_INTERVAL,
    DEFAULT_ZO_GPU_MEMORY_UTILIZATION,
    DEFAULT_ZO_GRADIENT_ACCUMULATION_UPDATE_STEPS,
    DEFAULT_ZO_LEARNING_RATE,
    DEFAULT_ZO_LOZO_PROVIDER_MODE,
    DEFAULT_ZO_MAX_NEW_TOKENS,
    DEFAULT_ZO_MODEL_NAME,
    DEFAULT_ZO_NUM_DEV,
    DEFAULT_ZO_NUM_SAMPLES,
    DEFAULT_ZO_NU,
    DEFAULT_ZO_PERTURBATION_NORMALIZATION,
    DEFAULT_ZO_PERTURB_EMBEDDINGS,
    DEFAULT_ZO_PROFILE_MODE,
    DEFAULT_ZO_QKV_WEIGHT_UPDATE,
    DEFAULT_ZO_QUANTIZED_UPDATE_MODE,
    DEFAULT_ZO_RANDOM_DEVICE,
    DEFAULT_ZO_RANK,
    DEFAULT_ZO_SCORING_BACKEND,
    DEFAULT_ZO_SEED,
    DEFAULT_ZO_STEPS,
    DEFAULT_ZO_SYNC_WEIGHT_UPDATE,
    DEFAULT_ZO_TASK_SHUFFLE_IMPL,
    DEFAULT_ZO_TRAIN_OBJECTIVE,
    DEFAULT_ZO_TRAIN_SAMPLER,
    DEFAULT_ZO_TRAIN_SCOPE,
    DEFAULT_ZO_U_BETA,
    DEFAULT_ZO_U_NORM_CAP,
    DEFAULT_ZO_UPDATE_BANK_RANK,
    DEFAULT_ZO_WANDB_PROJECT,
    DEFAULT_ZO_WARMUP_STEPS,
    DEFAULT_ZO_WEIGHT_UPDATE,
    DEFAULT_ZO_WEIGHT_UPDATE_PRECISION,
    DEFAULT_ZO_WEIGHT_DECAY,
)
from zo_vllm.core.lora_scope import LORA_TRAIN_SCOPE_CHOICES
from zo_vllm.tasks.tokenization import OPT_BOS_MODES, OPT_BOS_NATIVE


def build_vllm_zo_task_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=DEFAULT_ZO_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_ZO_WEIGHT_DECAY)
    parser.add_argument("--rank", type=int, default=DEFAULT_ZO_RANK)
    parser.add_argument(
        "--direction-scale",
        type=float,
        default=DEFAULT_ZO_DIRECTION_SCALE,
        help=("User amplitude multiplier applied after perturbation normalization."),
    )
    parser.add_argument(
        "--perturbation-normalization",
        choices=["rms", "none"],
        default=DEFAULT_ZO_PERTURBATION_NORMALIZATION,
        help=(
            "Apply analytic rank normalization using each provider's reference "
            "direction energy. Use none for legacy raw scale."
        ),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_ZO_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_ZO_WARMUP_STEPS)
    parser.add_argument("--eps", type=float, default=DEFAULT_ZO_EPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ZO_BATCH_SIZE)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_ZO_NUM_SAMPLES)
    parser.add_argument("--num-dev", type=int, default=DEFAULT_ZO_NUM_DEV)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_ZO_EVAL_INTERVAL)
    parser.add_argument("--eval-interval-epochs", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_ZO_SEED)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_ZO_DATA_SEED)
    parser.add_argument(
        "--train-sampler",
        choices=["sequential", "hf_random"],
        default=DEFAULT_ZO_TRAIN_SAMPLER,
    )
    parser.add_argument(
        "--task-shuffle-impl",
        choices=["numpy", "hf"],
        default=DEFAULT_ZO_TASK_SHUFFLE_IMPL,
        help=(
            "Dataset row shuffle implementation used before task batching. "
            "Use numpy for current Phase 3 runs; use hf only for exact "
            "comparison with older Hugging Face dataset.shuffle artifacts."
        ),
    )
    parser.add_argument("--dataloader-seed", type=int, default=None)
    parser.add_argument("--dataloader-drop-last", action="store_true")
    parser.add_argument("--train-objective", default=DEFAULT_ZO_TRAIN_OBJECTIVE)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config-name", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument(
        "--estimator",
        choices=["single_direction_antithetic", "multi_query"],
        default="single_direction_antithetic",
    )
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument(
        "--perturbation-sides",
        choices=["two_sided", "one_sided", "two-sided", "one-sided"],
        default="two_sided",
    )
    parser.add_argument("--query-microbatch-size", type=int, default=2)
    parser.add_argument(
        "--zo-random-device",
        choices=["cpu", "cuda"],
        default=DEFAULT_ZO_RANDOM_DEVICE,
    )
    parser.add_argument(
        "--direction-sampling",
        choices=["exact", "flat"],
        default=DEFAULT_ZO_DIRECTION_SAMPLING,
    )
    parser.add_argument(
        "--direction-provider",
        choices=["lozo", "agzo", "uagzo", "suagzo"],
        default=DEFAULT_ZO_DIRECTION_PROVIDER,
    )
    parser.add_argument(
        "--lozo-provider-mode",
        choices=["fast", "scheduled"],
        default=DEFAULT_ZO_LOZO_PROVIDER_MODE,
        help=(
            "LOZO direction provider implementation. fast preserves the calibrated "
            "Phase 3 path; scheduled uses the unified U/V provider stack."
        ),
    )
    parser.add_argument(
        "--nu",
        type=int,
        default=DEFAULT_ZO_NU,
        help=(
            "Basis/subspace reuse length. LOZO reuses random V; AGZO reuses the "
            "activation-guided subspace."
        ),
    )
    parser.add_argument(
        "--kappa", "--agzo-kappa", dest="agzo_kappa", type=int, default=1
    )
    parser.add_argument("--agzo-power-iter-steps", type=int, default=5)
    parser.add_argument("--agzo-low-rank-oversample", type=int, default=4)
    parser.add_argument(
        "--agzo-basis-method",
        choices=["power_iter", "svd", "low_rank_svd"],
        default="power_iter",
    )
    parser.add_argument(
        "--agzo-activation-force-eager", choices=["0", "1"], default="1"
    )
    parser.add_argument("--u-dim", type=int, default=None)
    parser.add_argument(
        "--train-scope",
        choices=list(LORA_TRAIN_SCOPE_CHOICES),
        default=DEFAULT_ZO_TRAIN_SCOPE,
    )
    parser.add_argument(
        "--perturb-embeddings",
        choices=["0", "1"],
        default=DEFAULT_ZO_PERTURB_EMBEDDINGS,
        help="Also perturb token embedding matrices through vLLM embedding LoRA.",
    )
    parser.add_argument(
        "--enforce-eager", choices=["0", "1"], default=DEFAULT_ZO_ENFORCE_EAGER
    )
    parser.add_argument(
        "--weight-update",
        choices=["direct"],
        default=DEFAULT_ZO_WEIGHT_UPDATE,
    )
    parser.add_argument(
        "--weight-update-precision",
        choices=["float32", "param"],
        default=DEFAULT_ZO_WEIGHT_UPDATE_PRECISION,
    )
    parser.add_argument(
        "--direct-update-mode",
        choices=["immediate", "accumulate"],
        default=DEFAULT_ZO_DIRECT_UPDATE_MODE,
    )
    parser.add_argument(
        "--quantized-update-mode",
        choices=["none", "lora_bank"],
        default=DEFAULT_ZO_QUANTIZED_UPDATE_MODE,
    )
    parser.add_argument("--update-bank-rank", default=DEFAULT_ZO_UPDATE_BANK_RANK)
    parser.add_argument(
        "--vllm-quantization",
        default=None,
        help="Optional vLLM quantization mode, for example fp8_per_tensor.",
    )
    parser.add_argument(
        "--prequant-model",
        default=None,
        help=(
            "Optional pre-quantized model path/name for Phase7 smoke tests. "
            "If unset, PHASE7_PREQUANT_MODEL is used when present."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-update-steps",
        type=int,
        default=DEFAULT_ZO_GRADIENT_ACCUMULATION_UPDATE_STEPS,
        help=(
            "When using accumulated direct updates, keep per-step low-rank "
            "updates in a pending buffer and flush them into the accumulated "
            "update LoRA every N measured steps. Use 0 to update the "
            "accumulated LoRA every step."
        ),
    )
    parser.add_argument(
        "--u-beta",
        type=float,
        default=DEFAULT_ZO_U_BETA,
        help=(
            "Per-step retention factor applied to accumulated U before adding "
            "the current projected update."
        ),
    )
    parser.add_argument(
        "--u-norm-cap",
        type=float,
        default=DEFAULT_ZO_U_NORM_CAP,
        help="Global L2 norm cap for accumulated U after each update.",
    )
    parser.add_argument(
        "--qkv-weight-update",
        choices=["separate", "batched"],
        default=DEFAULT_ZO_QKV_WEIGHT_UPDATE,
    )
    parser.add_argument(
        "--sync-weight-update",
        choices=["0", "1"],
        default=DEFAULT_ZO_SYNC_WEIGHT_UPDATE,
    )
    parser.add_argument(
        "--scoring-backend",
        choices=["generate", "direct_worker"],
        default=DEFAULT_ZO_SCORING_BACKEND,
    )
    parser.add_argument(
        "--direct-worker-max-logits-tokens",
        type=int,
        default=DEFAULT_ZO_DIRECT_WORKER_MAX_LOGITS_TOKENS,
    )
    parser.add_argument(
        "--score-chunk-size",
        type=int,
        default=0,
        help=(
            "Split plus/minus training scoring into chunks of this many rows. "
            "Use 0 to score the whole batch at once."
        ),
    )
    parser.add_argument(
        "--direct-worker-loss-impl",
        choices=["logprobs", "cross_entropy"],
        default=DEFAULT_ZO_DIRECT_WORKER_LOSS_IMPL,
    )
    parser.add_argument(
        "--base-eval-mode",
        choices=["generate", "direct_worker", "skip"],
        default=DEFAULT_ZO_BASE_EVAL_MODE,
    )
    parser.add_argument(
        "--profile-mode",
        choices=["minimal", "detailed"],
        default=DEFAULT_ZO_PROFILE_MODE,
    )
    parser.add_argument("--direction-digest", action="store_true")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--trace-step-events", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--train-loss-interval", type=int, default=0)
    parser.add_argument("--progress-interval-epochs", type=float, default=0.0)
    parser.add_argument("--train-loss-interval-epochs", type=float, default=0.0)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_ZO_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_ZO_MODEL_NAME)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument(
        "--opt-bos-mode",
        choices=list(OPT_BOS_MODES),
        default=OPT_BOS_NATIVE,
        help=(
            "OPT tokenizer BOS policy. native keeps HuggingFace/vLLM OPT "
            "default BOS=</s> id 2. lozo uses <s> id 0 for strict LOZO/MeZO "
            "baseline reproduction."
        ),
    )
    parser.add_argument(
        "--direct-lora-from-directions",
        choices=["0", "1"],
        default=DEFAULT_ZO_DIRECT_LORA_FROM_DIRECTIONS,
    )
    parser.add_argument(
        "--save-strategy",
        choices=["no", "steps", "best"],
        default="steps",
        help=(
            "HF-like checkpoint save strategy. no disables runtime checkpoints; "
            "steps saves every --save-steps measured steps; best saves only when "
            "metric-for-best-model improves at evaluation time."
        ),
    )
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--save-checkpoint-mode",
        choices=["auto", "metadata", "native", "lora"],
        default="auto",
        help=(
            "Checkpoint payload to save at save intervals. auto writes native "
            "checkpoints when load-best-model-at-end is enabled, otherwise "
            "metadata-only checkpoints. lora writes LoRA bank state only and "
            "currently supports --quantized-update-mode lora_bank."
        ),
    )
    parser.add_argument(
        "--load-best-model-at-end",
        choices=["0", "1"],
        default="0",
        help=(
            "HF Trainer-style behavior: track metric-for-best-model and load the "
            "best native checkpoint before final evaluation."
        ),
    )
    parser.add_argument(
        "--metric-for-best-model",
        default="eval_loss",
        help="Eval metric used to choose the best checkpoint.",
    )
    parser.add_argument(
        "--greater-is-better",
        choices=["auto", "0", "1"],
        default="auto",
        help="Whether metric-for-best-model should be maximized.",
    )
    parser.add_argument(
        "--save-final-checkpoint",
        choices=["0", "1"],
        default="0",
        help=(
            "Save a final vLLM sharded model checkpoint containing the effective "
            "model weights."
        ),
    )
    parser.add_argument(
        "--resume-lora-checkpoint",
        default=None,
        help=(
            "Resume from a lightweight LoRA bank checkpoint produced by "
            "--save-checkpoint-mode lora. Requires lora_bank update mode; "
            "AGZO/UAGZO/SUAGZO resume stores the active combined V block."
        ),
    )
    parser.add_argument("--u-snapshot-interval", type=int, default=0)
    parser.add_argument(
        "--u-snapshot-dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--u-snapshot-total-limit", type=int, default=0)
    parser.add_argument(
        "--eval-accuracy-samples",
        type=int,
        default=DEFAULT_ZO_EVAL_ACCURACY_SAMPLES,
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_ZO_MAX_NEW_TOKENS)
    parser.add_argument(
        "--accuracy-eval-mode",
        choices=["auto", "full", "skip"],
        default=DEFAULT_ZO_ACCURACY_EVAL_MODE,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root for auto-created run directories when --output-dir is not set. "
            "AGZO/UAGZO defaults to zo_post/results; LOZO defaults to phase3/results."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Run directory name used with --output-root when --output-dir is not set.",
    )
    parser.add_argument("--report-to", default="")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default=DEFAULT_ZO_WANDB_PROJECT)
    return parser
