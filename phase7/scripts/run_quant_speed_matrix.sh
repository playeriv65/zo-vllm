#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/phase7/scripts/common.sh"
cd "${ROOT_DIR}"

GPU="${GPU:-}"
phase7_require_gpu

MODELS="${MODELS:-opt13b opt13b_gptq}"
EAGER_MODES="${EAGER_MODES:-0}"
BATCH_SIZES="${BATCH_SIZES:-64}"
STEPS="${STEPS:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
RANK="${RANK:-2}"
NU="${NU:-50}"
LR="${LR:-1e-7}"
EPS="${EPS:-1e-3}"
UPDATE_BANK_RANK="${UPDATE_BANK_RANK:-64}"
NUM_DEV="${NUM_DEV:-128}"
EVAL_ACCURACY_SAMPLES="${EVAL_ACCURACY_SAMPLES:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-0}"
PROFILE_MODE="${PROFILE_MODE:-minimal}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="${SUMMARY_LOG:-${ROOT_DIR}/phase7/logs/phase7_quant_speed_matrix_${RUN_TS}.log}"

mkdir -p "$(dirname "${SUMMARY_LOG}")"

phase7_export_runtime_env

{
  echo "[phase7-speed] run_ts=${RUN_TS}"
  echo "[phase7-speed] gpu=${GPU}"
  echo "[phase7-speed] models=${MODELS}"
  echo "[phase7-speed] eager_modes=${EAGER_MODES}"
  echo "[phase7-speed] batch_sizes=${BATCH_SIZES}"
  echo "[phase7-speed] steps=${STEPS} warmup_steps=${WARMUP_STEPS}"
  echo "[phase7-speed] rank=${RANK} nu=${NU} lr=${LR} eps=${EPS}"
  echo "[phase7-speed] update_bank_rank=${UPDATE_BANK_RANK}"
  echo "[phase7-speed] max_model_len=${MAX_MODEL_LEN} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  echo "[phase7-speed] report_to=${REPORT_TO}"
} | tee "${SUMMARY_LOG}"

for model_key in ${MODELS}; do
  phase7_model_config "${model_key}"
  MODEL_NAME="${PHASE7_MODEL_NAME}"
  PREQUANT_MODEL="${PHASE7_PREQUANT_MODEL}"
  VLLM_QUANTIZATION="${PHASE7_VLLM_QUANTIZATION}"
  for enforce_eager in ${EAGER_MODES}; do
    for batch_size in ${BATCH_SIZES}; do
      max_num_seqs="${MAX_NUM_SEQS:-$((batch_size * 4))}"
      num_samples="${NUM_SAMPLES:-$(((STEPS + WARMUP_STEPS) * batch_size))}"
      run_id="phase7_speed_${model_key}_b${batch_size}_eager${enforce_eager}_steps${STEPS}_${RUN_TS}"
      run_dir="${ROOT_DIR}/phase7/logs/${run_id}"
      log_file="${run_dir}/run.log"
      mkdir -p "${run_dir}"

      phase7_model_args "${PREQUANT_MODEL}" "${VLLM_QUANTIZATION}"

      {
        echo
        echo "[phase7-speed] run_id=${run_id}"
        echo "[phase7-speed] model_key=${model_key}"
        echo "[phase7-speed] model_name=${MODEL_NAME}"
        echo "[phase7-speed] prequant_model=${PREQUANT_MODEL:-none}"
        echo "[phase7-speed] vllm_quantization=${VLLM_QUANTIZATION:-none}"
        echo "[phase7-speed] batch_size=${batch_size} num_samples=${num_samples}"
        echo "[phase7-speed] enforce_eager=${enforce_eager}"
        echo "[phase7-speed] max_num_seqs=${max_num_seqs}"
        echo "[phase7-speed] output_dir=${run_dir}/vllm"
      } | tee -a "${SUMMARY_LOG}" | tee "${log_file}"

      .venv/bin/python -u -m zo_vllm.experiment.runners.vllm_zo_task \
        --model-name "${MODEL_NAME}" \
        "${PHASE7_EXTRA_MODEL_ARGS[@]}" \
        --steps "${STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --rank "${RANK}" \
        --nu "${NU}" \
        --lr "${LR}" \
        --eps "${EPS}" \
        --batch-size "${batch_size}" \
        --num-samples "${num_samples}" \
        --num-dev "${NUM_DEV}" \
        --eval-interval 0 \
        --eval-accuracy-samples "${EVAL_ACCURACY_SAMPLES}" \
        --seed "${SEED}" \
        --data-seed "${DATA_SEED}" \
        --enforce-eager "${enforce_eager}" \
        --quantized-update-mode lora_bank \
        --update-bank-rank "${UPDATE_BANK_RANK}" \
        --base-eval-mode skip \
        --accuracy-eval-mode skip \
        --profile-mode "${PROFILE_MODE}" \
        --progress-interval 20 \
        --train-loss-interval 0 \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${max_num_seqs}" \
        --output-dir "${run_dir}/vllm" \
        --report-to "${REPORT_TO}" \
        --run-name "${run_id}" \
        --wandb-project zo-vllm \
        2>&1 | tee -a "${log_file}"

      json_path="$(ls -1t "${run_dir}"/vllm/vllm_perf_*.json | head -n 1)"
      .venv/bin/python - <<'PY' "${json_path}" "${run_id}" "${SUMMARY_LOG}"
import json
import sys

json_path, run_id, summary_log = sys.argv[1:4]
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
timing = data["timing"]
line = (
    f"[phase7-speed-result] run_id={run_id} "
    f"json={json_path} "
    f"mean_step_s={timing['step_s']['mean']:.6f} "
    f"tail100_step_s={timing['tail_100']['step_s']['mean']:.6f} "
    f"mean_score_s={timing['score_s']['mean']:.6f} "
    f"tail100_score_s={timing['tail_100']['score_s']['mean']:.6f} "
    f"mean_update_s={timing['aligned_phase_s']['update_s']['mean']:.6f} "
    f"tail100_update_s={timing['aligned_phase_s']['tail_100']['update_s']['mean']:.6f}"
)
print(line)
with open(summary_log, "a", encoding="utf-8") as f:
    f.write(line + "\n")
PY
    done
  done
done
