#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/phase7/scripts/common.sh"
cd "${ROOT_DIR}"

GPU="${GPU:-}"
phase7_require_gpu

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-20000}"
RANK="${RANK:-2}"
NU="${NU:-50}"
LR="${LR:-1e-7}"
EPS="${EPS:-1e-3}"
UPDATE_BANK_RANK="${UPDATE_BANK_RANK:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
REPORT_TO="${REPORT_TO:-wandb}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-20}"
PROFILE_MODE="${PROFILE_MODE:-minimal}"

phase7_export_runtime_env

run_case() {
  local case_key="$1"
  local model_key="qwen3_8b"
  local quantized_update_mode="none"
  local run_tag="${case_key}"

  case "${case_key}" in
    bf16_direct)
      quantized_update_mode="none"
      ;;
    bf16_bank)
      quantized_update_mode="lora_bank"
      ;;
    fp8_bank)
      model_key="qwen3_8b_fp8"
      quantized_update_mode="lora_bank"
      ;;
    *)
      echo "Unknown case: ${case_key}" >&2
      exit 2
      ;;
  esac
  phase7_model_config "${model_key}"
  local model_name="${PHASE7_MODEL_NAME}"
  local prequant_model="${PHASE7_PREQUANT_MODEL}"
  local vllm_quantization="${PHASE7_VLLM_QUANTIZATION}"

  local run_id="phase7_mode_compare_${run_tag}_b${BATCH_SIZE}_r${RANK}_nu${NU}_steps${STEPS}_warm${WARMUP_STEPS}_${RUN_TS}"
  local run_dir="${ROOT_DIR}/phase7/logs/${run_id}"
  mkdir -p "${run_dir}"

  {
    echo "[phase7-compare] run_id=${run_id}"
    echo "[phase7-compare] case=${case_key}"
    echo "[phase7-compare] model_name=${model_name}"
    echo "[phase7-compare] prequant_model=${prequant_model:-none}"
    echo "[phase7-compare] quantized_update_mode=${quantized_update_mode}"
    echo "[phase7-compare] gpu=${GPU}"
    echo "[phase7-compare] steps=${STEPS} warmup_steps=${WARMUP_STEPS} batch_size=${BATCH_SIZE} num_samples=${NUM_SAMPLES}"
    echo "[phase7-compare] rank=${RANK} nu=${NU} lr=${LR} eps=${EPS} update_bank_rank=${UPDATE_BANK_RANK}"
    echo "[phase7-compare] enforce_eager=${ENFORCE_EAGER} profile_mode=${PROFILE_MODE}"
    echo "[phase7-compare] output_dir=${run_dir}/vllm"
  } | tee "${run_dir}/run.log"

  phase7_model_args "${prequant_model}" "${vllm_quantization}"
  if [[ "${quantized_update_mode}" == "lora_bank" ]]; then
    PHASE7_EXTRA_MODEL_ARGS+=(--quantized-update-mode lora_bank --update-bank-rank "${UPDATE_BANK_RANK}")
  else
    PHASE7_EXTRA_MODEL_ARGS+=(--quantized-update-mode none --update-bank-rank "${UPDATE_BANK_RANK}")
  fi

  .venv/bin/python -u -m zo_vllm.experiment.runners.vllm_zo_task \
    --model-name "${model_name}" \
    "${PHASE7_EXTRA_MODEL_ARGS[@]}" \
    --steps "${STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --rank "${RANK}" \
    --nu "${NU}" \
    --lr "${LR}" \
    --eps "${EPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-dev 500 \
    --eval-interval 0 \
    --eval-accuracy-samples 500 \
    --seed 42 \
    --data-seed 0 \
    --enforce-eager "${ENFORCE_EAGER}" \
    --base-eval-mode skip \
    --profile-mode "${PROFILE_MODE}" \
    --record-history \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --train-loss-interval 0 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --accuracy-eval-mode skip \
    --output-dir "${run_dir}/vllm" \
    --report-to "${REPORT_TO}" \
    --run-name "${run_id}" \
    --wandb-project zo-vllm \
    2>&1 | tee -a "${run_dir}/run.log"
}

if [[ "$#" -gt 0 ]]; then
  for case_key in "$@"; do
    run_case "${case_key}"
  done
else
  run_case bf16_direct
  run_case bf16_bank
  run_case fp8_bank
fi
