#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/phase7/scripts/common.sh"

MODEL_KEY="${1:-${MODEL_KEY:-opt13b}}"

phase7_model_config "${MODEL_KEY}"
cd "${ROOT_DIR}"

MODEL_NAME="${MODEL_NAME:-${PHASE7_MODEL_NAME}}"
PREQUANT_MODEL="${PREQUANT_MODEL:-${PHASE7_PREQUANT_MODEL}}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-${PHASE7_VLLM_QUANTIZATION}}"
GPU="${GPU:-}"
STEPS="${STEPS:-200}"
RANK="${RANK:-2}"
NU="${NU:-50}"
LR="${LR:-1e-7}"
EPS="${EPS:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-3200}"
NUM_DEV="${NUM_DEV:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_ACCURACY_SAMPLES="${EVAL_ACCURACY_SAMPLES:-500}"
UPDATE_BANK_RANK="${UPDATE_BANK_RANK:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-phase7_mid_${PHASE7_MODEL_TAG}_r${RANK}_nu${NU}_lr${LR}_eps${EPS}_steps${STEPS}_${RUN_TS}}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/phase7/logs/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/run.log}"

mkdir -p "${RUN_DIR}"

phase7_require_gpu
phase7_export_runtime_env
phase7_model_args "${PREQUANT_MODEL}" "${VLLM_QUANTIZATION}"

{
  echo "[phase7] run_id=${RUN_ID}"
  echo "[phase7] model_key=${MODEL_KEY}"
  echo "[phase7] model_name=${MODEL_NAME}"
  echo "[phase7] prequant_model=${PREQUANT_MODEL:-none}"
  echo "[phase7] vllm_quantization=${VLLM_QUANTIZATION:-none}"
  echo "[phase7] gpu=${GPU}"
  echo "[phase7] steps=${STEPS} batch_size=${BATCH_SIZE} num_samples=${NUM_SAMPLES}"
  echo "[phase7] rank=${RANK} nu=${NU} lr=${LR} eps=${EPS}"
  echo "[phase7] quantized_update_mode=lora_bank update_bank_rank=${UPDATE_BANK_RANK}"
  echo "[phase7] eval_interval=${EVAL_INTERVAL} num_dev=${NUM_DEV} eval_accuracy_samples=${EVAL_ACCURACY_SAMPLES}"
  echo "[phase7] seed=${SEED} data_seed=${DATA_SEED}"
  echo "[phase7] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN}"
  echo "[phase7] max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} max_num_seqs=${MAX_NUM_SEQS}"
  echo "[phase7] enforce_eager=${ENFORCE_EAGER}"
  echo "[phase7] output_dir=${RUN_DIR}/vllm"
} | tee "${LOG_FILE}"

.venv/bin/python -u -m zo_vllm.experiment.runners.vllm_zo_task \
  --model-name "${MODEL_NAME}" \
  "${PHASE7_EXTRA_MODEL_ARGS[@]}" \
  --steps "${STEPS}" \
  --rank "${RANK}" \
  --nu "${NU}" \
  --lr "${LR}" \
  --eps "${EPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-samples "${NUM_SAMPLES}" \
  --num-dev "${NUM_DEV}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-accuracy-samples "${EVAL_ACCURACY_SAMPLES}" \
  --seed "${SEED}" \
  --data-seed "${DATA_SEED}" \
  --enforce-eager "${ENFORCE_EAGER}" \
  --quantized-update-mode lora_bank \
  --update-bank-rank "${UPDATE_BANK_RANK}" \
  --base-eval-mode direct_worker \
  --profile-mode minimal \
  --record-history \
  --progress-interval 10 \
  --train-loss-interval 0 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --accuracy-eval-mode full \
  --output-dir "${RUN_DIR}/vllm" \
  --report-to wandb \
  --run-name "${RUN_ID}" \
  --wandb-project zo-vllm \
  2>&1 | tee -a "${LOG_FILE}"
