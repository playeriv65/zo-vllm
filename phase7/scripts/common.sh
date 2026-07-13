#!/usr/bin/env bash

phase7_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

phase7_model_usage() {
  echo "Use opt13b, opt13b_gptq, qwen3_8b, or qwen3_8b_fp8."
}

phase7_model_config() {
  case "$1" in
    opt13b)
      PHASE7_MODEL_NAME="facebook/opt-13b"
      PHASE7_PREQUANT_MODEL=""
      PHASE7_VLLM_QUANTIZATION=""
      PHASE7_MODEL_TAG="opt13b"
      ;;
    opt13b_gptq)
      PHASE7_MODEL_NAME="facebook/opt-13b"
      PHASE7_PREQUANT_MODEL="iproskurina/opt-13b-GPTQ-4bit-g128"
      PHASE7_VLLM_QUANTIZATION="gptq"
      PHASE7_MODEL_TAG="opt13b_gptq"
      ;;
    qwen3_8b)
      PHASE7_MODEL_NAME="Qwen/Qwen3-8B"
      PHASE7_PREQUANT_MODEL=""
      PHASE7_VLLM_QUANTIZATION=""
      PHASE7_MODEL_TAG="qwen3_8b"
      ;;
    qwen3_8b_fp8)
      PHASE7_MODEL_NAME="Qwen/Qwen3-8B"
      PHASE7_PREQUANT_MODEL="Qwen/Qwen3-8B-FP8"
      PHASE7_VLLM_QUANTIZATION=""
      PHASE7_MODEL_TAG="qwen3_8b_fp8"
      ;;
    *)
      echo "Unknown model key: $1. $(phase7_model_usage)" >&2
      return 2
      ;;
  esac
}

phase7_require_gpu() {
  if [[ -z "${GPU:-}" ]]; then
    echo "Set GPU=<visible GPU index> before running Phase7 experiments." >&2
    return 2
  fi
}

phase7_export_runtime_env() {
  export CUDA_VISIBLE_DEVICES="${GPU}"
  export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
  export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
  export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
}

phase7_model_args() {
  local prequant_model="$1"
  local vllm_quantization="$2"
  PHASE7_EXTRA_MODEL_ARGS=()
  if [[ -n "${prequant_model}" ]]; then
    PHASE7_EXTRA_MODEL_ARGS+=(--prequant-model "${prequant_model}")
  fi
  if [[ -n "${vllm_quantization}" ]]; then
    PHASE7_EXTRA_MODEL_ARGS+=(--vllm-quantization "${vllm_quantization}")
  fi
}
