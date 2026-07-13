#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <gpu_id> <full|skip_1d|skip_pos|skip_1d_pos>" >&2
  exit 2
fi

GPU_ID="$1"
SCOPE="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/phase6/results/opt13b_mezo_scope_ablation_${RUN_TS}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/phase6/logs/opt13b_mezo_scope_ablation_${RUN_TS}}"

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

extra_flags=()
case "${SCOPE}" in
  full) ;;
  skip_1d) extra_flags+=(--mezo_skip_1d) ;;
  skip_pos) extra_flags+=(--mezo_skip_position_embeddings) ;;
  skip_1d_pos) extra_flags+=(--mezo_skip_1d --mezo_skip_position_embeddings) ;;
  *)
    echo "unknown scope: ${SCOPE}" >&2
    exit 2
    ;;
esac

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

run_name="phase6_opt13b_mezo_${SCOPE}_s${STEPS}_b${BATCH_SIZE}_lr${LR}_eps${EPS}_${RUN_TS}"
run_dir="${RESULT_ROOT}/${run_name}"
log_file="${LOG_ROOT}/${run_name}.log"
mkdir -p "${run_dir}"

(
  cd "${PROJECT_ROOT}/third_party/LOZO/large_models"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export WANDB_PROJECT
  export WANDB_ENTITY
  export WANDB_NAME="${run_name}"
  {
    echo "[phase6-mezo-scope] started_at=$(date --iso-8601=seconds)"
    echo "[phase6-mezo-scope] cwd=$(pwd)"
    echo "[phase6-mezo-scope] gpu_id=${GPU_ID}"
    echo "[phase6-mezo-scope] scope=${SCOPE}"
    echo "[phase6-mezo-scope] result_dir=${run_dir}"
    echo "[phase6-mezo-scope] log_file=${log_file}"
    echo "[phase6-mezo-scope] flags=${extra_flags[*]:-none}"
    echo "[phase6-mezo-scope] command=.venv/bin/python -u run_mezo.py --model_name ${MODEL_NAME} --task_name ${TASK_NAME} --max_steps ${STEPS} --per_device_train_batch_size ${BATCH_SIZE} --learning_rate ${LR} --zo_eps ${EPS} --eval_steps ${EVAL_INTERVAL} ${extra_flags[*]:-}"
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
      "${extra_flags[@]}"
    echo "[phase6-mezo-scope] finished_at=$(date --iso-8601=seconds)"
  } 2>&1 | tee "${log_file}"
)
