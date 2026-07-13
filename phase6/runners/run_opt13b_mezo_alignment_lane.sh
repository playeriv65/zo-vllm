#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <gpu0|gpu1>" >&2
  exit 2
fi

LANE="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/phase6/results/opt13b_mezo_alignment_${RUN_TS}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/phase6/logs/opt13b_mezo_alignment_${RUN_TS}}"

MODEL_NAME="${MODEL_NAME:-facebook/opt-13b}"
TASK_NAME="${TASK_NAME:-SST2}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_DEV="${NUM_DEV:-500}"
NUM_EVAL="${NUM_EVAL:-872}"
SEED="${SEED:-42}"
TRAIN_SET_SEED="${TRAIN_SET_SEED:-0}"
LR="${LR:-1e-7}"
EPS="${EPS:-1e-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-lozo-vllm-phase6}"
WANDB_ENTITY="${WANDB_ENTITY:-playeriv65-university-of-minnesota}"
OPT_BOS_MODE="${OPT_BOS_MODE:-lozo}"

VLLM_NU="${VLLM_NU:-1}"
VLLM_PERTURBATION_NORMALIZATION="${VLLM_PERTURBATION_NORMALIZATION:-none}"
VLLM_DIRECTION_SCALE="${VLLM_DIRECTION_SCALE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

log_and_run() {
  local log_file="$1"
  shift
  {
    echo "[phase6-opt13b] started_at=$(date --iso-8601=seconds)"
    echo "[phase6-opt13b] cwd=$(pwd)"
    echo "[phase6-opt13b] command=$*"
    "$@"
    echo "[phase6-opt13b] finished_at=$(date --iso-8601=seconds)"
  } 2>&1 | tee "${log_file}"
}

run_mezo() {
  local gpu_id="$1"
  local scope="$2"
  local extra_flag=()
  if [[ "${scope}" == "lora_normal" ]]; then
    extra_flag=(--mezo_lora_normal)
  fi

  local run_name="phase6_opt13b_mezo_${scope}_s${STEPS}_b${BATCH_SIZE}_lr${LR}_eps${EPS}_${RUN_TS}"
  local run_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  mkdir -p "${run_dir}"

  (
    cd "${PROJECT_ROOT}/third_party/LOZO/large_models"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export WANDB_PROJECT
    export WANDB_ENTITY
    export WANDB_NAME="${run_name}"
    log_and_run "${log_file}" \
      .venv/bin/python -u run_mezo.py \
        --model_name "${MODEL_NAME}" \
        --task_name "${TASK_NAME}" \
        --output_dir "${run_dir}/hf_output" \
        --result_file "${run_dir}/metrics.json" \
        --tag "${run_name}" \
        --seed "${SEED}" \
        --train_set_seed "${TRAIN_SET_SEED}" \
        --num_train "${NUM_TRAIN}" \
        --num_dev "${NUM_DEV}" \
        --num_eval "${NUM_EVAL}" \
        --logging_steps 10 \
        --max_steps "${STEPS}" \
        --trainer zo \
        --load_float16 \
        --learning_rate "${LR}" \
        --zo_eps "${EPS}" \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --lr_scheduler_type constant \
        --evaluation_strategy steps \
        --eval_steps "${EVAL_INTERVAL}" \
        --save_strategy no \
        --save_total_limit 1 \
        --train_as_classification \
        --report_to wandb \
        --run_name "${run_name}" \
        "${extra_flag[@]}"
  )
}

run_vllm() {
  local gpu_id="$1"
  local rank="$2"
  local direction_scale="${VLLM_DIRECTION_SCALE}"
  if [[ "${direction_scale}" == "auto" ]]; then
    direction_scale="$(python - <<PY
import math
print(1.0 / math.sqrt(${rank}))
PY
)"
  fi
  local run_name="phase6_opt13b_vllm_r${rank}_nu${VLLM_NU}_s${STEPS}_b${BATCH_SIZE}_lr${LR}_eps${EPS}_scale${direction_scale}_bos${OPT_BOS_MODE}_${RUN_TS}"
  local run_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  mkdir -p "${run_dir}"

  (
    cd "${PROJECT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export WANDB_PROJECT
    export WANDB_ENTITY
    log_and_run "${log_file}" \
      .venv/bin/python -u -m zo_vllm.experiment.runners.vllm_zo_task \
        --model-name "${MODEL_NAME}" \
        --opt-bos-mode "${OPT_BOS_MODE}" \
        --train-objective sst2_classification \
        --steps "${STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-samples "${NUM_TRAIN}" \
        --num-dev "${NUM_DEV}" \
        --eval-accuracy-samples "${NUM_EVAL}" \
        --eval-interval "${EVAL_INTERVAL}" \
        --seed "${SEED}" \
        --data-seed "${TRAIN_SET_SEED}" \
        --dataloader-seed "${SEED}" \
        --lr "${LR}" \
        --eps "${EPS}" \
        --rank "${rank}" \
        --nu "${VLLM_NU}" \
        --direction-scale "${direction_scale}" \
        --perturbation-normalization "${VLLM_PERTURBATION_NORMALIZATION}" \
        --direction-sampling flat \
        --train-scope lora_normal \
        --batch-invariant 0 \
        --enforce-eager 0 \
        --lora-residency gpu \
        --lora-injection direct \
        --weight-update direct \
        --weight-update-precision param \
        --direct-update-mode immediate \
        --qkv-weight-update batched \
        --sync-weight-update 0 \
        --scoring-backend direct_worker \
        --direct-worker-loss-impl logprobs \
        --base-eval-mode direct_worker \
        --profile-mode minimal \
        --direct-lora-from-directions 1 \
        --train-loss-interval 10 \
        --progress-interval 100 \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --accuracy-eval-mode full \
        --output-dir "${run_dir}" \
        --report-to wandb \
        --wandb-project "${WANDB_PROJECT}" \
        --run-name "${run_name}"
  )
}

case "${LANE}" in
  gpu0)
    run_mezo 0 full
    run_vllm 0 128
    run_vllm 0 512
    ;;
  gpu1)
    run_mezo 1 lora_normal
    run_vllm 1 256
    ;;
  *)
    echo "unknown lane: ${LANE}" >&2
    exit 2
    ;;
esac
