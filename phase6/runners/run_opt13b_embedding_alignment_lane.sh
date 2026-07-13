#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <rank> [rank ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
RANKS=("$@")

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/phase6/results/opt13b_lora_full_${RUN_TS}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/phase6/logs/opt13b_lora_full_${RUN_TS}}"

MODEL_NAME="${MODEL_NAME:-facebook/opt-13b}"
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
WANDB_MODE="${WANDB_MODE:-online}"
OPT_BOS_MODE="${OPT_BOS_MODE:-lozo}"
VLLM_NU="${VLLM_NU:-1}"
VLLM_PERTURBATION_NORMALIZATION="${VLLM_PERTURBATION_NORMALIZATION:-none}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

direction_scale_for_rank() {
  local rank="$1"
  "${PROJECT_ROOT}/.venv/bin/python" - <<PY
import math
print(1.0 / math.sqrt(${rank}))
PY
}

run_rank() {
  local rank="$1"
  local direction_scale
  direction_scale="$(direction_scale_for_rank "${rank}")"
  local run_name="phase6_opt13b_vllm_lora_full_r${rank}_nu${VLLM_NU}_s${STEPS}_b${BATCH_SIZE}_lr${LR}_eps${EPS}_scale${direction_scale}_bos${OPT_BOS_MODE}_gpu${GPU_ID}_${RUN_TS}"
  local run_dir="${RESULT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  mkdir -p "${run_dir}"

  (
    cd "${PROJECT_ROOT}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export WANDB_PROJECT
    export WANDB_ENTITY
    export WANDB_MODE
    export PYTHONUNBUFFERED=1
    {
      echo "[phase6-opt13b-lora-full] started_at=$(date --iso-8601=seconds)"
      echo "[phase6-opt13b-lora-full] cwd=$(pwd)"
      echo "[phase6-opt13b-lora-full] gpu_id=${GPU_ID}"
      echo "[phase6-opt13b-lora-full] rank=${rank}"
      echo "[phase6-opt13b-lora-full] train_scope=lora_full"
      echo "[phase6-opt13b-lora-full] tied_lm_head=auto_when_model_config_tie_word_embeddings"
      echo "[phase6-opt13b-lora-full] result_dir=${run_dir}"
      echo "[phase6-opt13b-lora-full] log_file=${log_file}"
      echo "[phase6-opt13b-lora-full] command=.venv/bin/python -u -m zo_vllm.experiment.runners.vllm_zo_task --model-name ${MODEL_NAME} --opt-bos-mode ${OPT_BOS_MODE} --train-objective sst2_classification --steps ${STEPS} --batch-size ${BATCH_SIZE} --num-samples ${NUM_TRAIN} --num-dev ${NUM_DEV} --eval-accuracy-samples ${NUM_EVAL} --eval-interval ${EVAL_INTERVAL} --seed ${SEED} --data-seed ${TRAIN_SET_SEED} --dataloader-seed ${SEED} --lr ${LR} --eps ${EPS} --rank ${rank} --nu ${VLLM_NU} --direction-scale ${direction_scale} --perturbation-normalization ${VLLM_PERTURBATION_NORMALIZATION} --direction-sampling flat --direction-provider lozo --lozo-provider-mode fast --train-scope lora_full --enforce-eager 0 --weight-update direct --weight-update-precision param --direct-update-mode immediate --qkv-weight-update batched --sync-weight-update 0 --scoring-backend direct_worker --direct-worker-loss-impl logprobs --base-eval-mode direct_worker --profile-mode minimal --direct-lora-from-directions 1 --train-loss-interval 10 --progress-interval 100 --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION} --accuracy-eval-mode full --save-strategy no --output-dir ${run_dir} --report-to wandb --wandb-project ${WANDB_PROJECT} --run-name ${run_name}"
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
        --direction-provider lozo \
        --lozo-provider-mode fast \
        --train-scope lora_full \
        --enforce-eager 0 \
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
        --save-strategy no \
        --output-dir "${run_dir}" \
        --report-to wandb \
        --wandb-project "${WANDB_PROJECT}" \
        --run-name "${run_name}"
      echo "[phase6-opt13b-lora-full] finished_at=$(date --iso-8601=seconds)"
    } 2>&1 | tee "${log_file}"
  )
}

for rank in "${RANKS[@]}"; do
  run_rank "${rank}"
done
