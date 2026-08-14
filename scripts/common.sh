#!/usr/bin/env bash
set -euo pipefail

_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPTS_DIR}/.." && pwd)}"
TRL_ROOT="${TRL_ROOT:-${PROJECT_ROOT}/trl}"

export PYTHONPATH="${PROJECT_ROOT}:${TRL_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"

STUDENT_MODEL="${STUDENT_MODEL:-${PROJECT_ROOT}/models/Qwen3-1.7B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-${PROJECT_ROOT}/models/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/train/deepscaler_conversation.jsonl}"
EVAL_DATA="${EVAL_DATA:-${PROJECT_ROOT}/data/eval/math_test.jsonl}"

BF16="${BF16:-True}"
STUDENT_DTYPE="${STUDENT_DTYPE:-float32}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_LOAD_IN_4BIT="${TEACHER_LOAD_IN_4BIT:-False}"
TEACHER_LOAD_IN_8BIT="${TEACHER_LOAD_IN_8BIT:-False}"
TEACHER_BNB_4BIT_QUANT_TYPE="${TEACHER_BNB_4BIT_QUANT_TYPE:-nf4}"
TEACHER_BNB_4BIT_COMPUTE_DTYPE="${TEACHER_BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
TEACHER_USE_BNB_NESTED_QUANT="${TEACHER_USE_BNB_NESTED_QUANT:-False}"
TEACHER_BNB_8BIT_THRESHOLD="${TEACHER_BNB_8BIT_THRESHOLD:-6.0}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_LENGTH="${MAX_LENGTH:-6144}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
MAX_STEPS="${MAX_STEPS:-1500}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
SEED="${SEED:-42}"
USE_VLLM="${USE_VLLM:-True}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
EVAL_STEPS="${EVAL_STEPS:-30}"
EVAL_TEST_SAMPLES="${EVAL_TEST_SAMPLES:-100}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_N="${EVAL_N:-1}"
EVAL_K="${EVAL_K:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
KL_MIX_MODE="${KL_MIX_MODE:-jsd}"
LOG_REWARD_CLIP_MIN="${LOG_REWARD_CLIP_MIN:--5.0}"
LOG_REWARD_CLIP_MAX="${LOG_REWARD_CLIP_MAX:-5.0}"
LOG_REWARD_TANH_TEMPERATURE="${LOG_REWARD_TANH_TEMPERATURE:-5.0}"
POWER_REWARD_ALPHA="${POWER_REWARD_ALPHA:-0.1}"
TOKEN_LOSS_NORMALIZATION="${TOKEN_LOSS_NORMALIZATION:-microbatch_mean}"
REWARD_NORMALIZATION="${REWARD_NORMALIZATION:-none}"
TERMINAL_STOP_TARGET="${TERMINAL_STOP_TARGET:-im_end}"
REWARD_POSITION_PLOT_STEPS="${REWARD_POSITION_PLOT_STEPS:-10}"
REWARD_POSITION_MAX_INDEX="${REWARD_POSITION_MAX_INDEX:-${MAX_COMPLETION_LENGTH}}"
REWARD_POSITION_SMOOTH_WINDOW="${REWARD_POSITION_SMOOTH_WINDOW:-16}"
REPORT_TO="${REPORT_TO:-swanlab}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export REPORT_TO SWANLAB_MODE

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/experiments}"


reward_value_tag() {
  local tag="$1"
  tag="${tag//-/m}"
  tag="${tag//./p}"
  tag="$(printf '%s' "${tag}" | tr -cs '[:alnum:]_' '_')"
  printf '%s' "${tag}"
}


append_reward_param_tag() {
  local extra_tag="$1"
  if [[ -z "${extra_tag}" ]]; then
    return
  fi
  if [[ -n "${reward_param_tag}" ]]; then
    reward_param_tag="${reward_param_tag}_${extra_tag}"
  else
    reward_param_tag="${extra_tag}"
  fi
}

model_path_tag() {
  local model_path="${1%/}"
  local tag="${model_path##*/}"
  if [[ -z "${tag}" ]]; then
    tag="unknown"
  fi
  tag="${tag// /_}"
  tag="$(printf '%s' "${tag}" | tr -cs '[:alnum:]._-' '_')"
  printf '%s' "${tag}"
}

run_opd_dynamics() {
  local reward_mode="$1"
  local exp_name="$2"
  local thinking_tag
  local student_tag
  local teacher_tag
  local exp_name_with_models
  local terminal_tag="terminal_${TERMINAL_STOP_TARGET}"
  local maxlen_tag="maxlen${MAX_COMPLETION_LENGTH}"
  local reward_param_tag=""

  case "${ENABLE_THINKING,,}" in
    true) thinking_tag="thinking" ;;
    *) thinking_tag="non_thinking" ;;
  esac

  student_tag="$(model_path_tag "${STUDENT_MODEL}")"
  teacher_tag="$(model_path_tag "${TEACHER_MODEL}")"
  exp_name_with_models="${exp_name}_${reward_mode}_student${student_tag}_teacher${teacher_tag}"

  case "${reward_mode}" in
    sampled_log_clip)
      append_reward_param_tag "clip$(reward_value_tag "${LOG_REWARD_CLIP_MIN}")_$(reward_value_tag "${LOG_REWARD_CLIP_MAX}")"
      ;;
    sampled_log_tanh)
      append_reward_param_tag "tanhT$(reward_value_tag "${LOG_REWARD_TANH_TEMPERATURE}")"
      ;;
    sampled_power_diff)
      append_reward_param_tag "alpha$(reward_value_tag "${POWER_REWARD_ALPHA}")"
      ;;
  esac

  if [[ "${TOKEN_LOSS_NORMALIZATION}" != "microbatch_mean" || "${REWARD_NORMALIZATION}" != "none" ]]; then
    append_reward_param_tag "loss_${TOKEN_LOSS_NORMALIZATION}_reward_${REWARD_NORMALIZATION}"
  fi
  if [[ "${TEACHER_LOAD_IN_4BIT,,}" == "true" ]]; then
    append_reward_param_tag "teacher4bit"
  elif [[ "${TEACHER_LOAD_IN_8BIT,,}" == "true" ]]; then
    append_reward_param_tag "teacher8bit"
  fi

  local output_dir
  local run_name
  if [[ -n "${reward_param_tag}" ]]; then
    output_dir="${OUTPUT_DIR_OVERRIDE:-${EXPERIMENT_ROOT}/${exp_name_with_models}/${thinking_tag}/${maxlen_tag}/${terminal_tag}/${reward_param_tag}/${RUN_TAG}/checkpoints}"
    run_name="${RUN_NAME_OVERRIDE:-${exp_name_with_models}_${thinking_tag}_${maxlen_tag}_${terminal_tag}_${reward_param_tag}_${RUN_TAG}}"
  else
    output_dir="${OUTPUT_DIR_OVERRIDE:-${EXPERIMENT_ROOT}/${exp_name_with_models}/${thinking_tag}/${maxlen_tag}/${terminal_tag}/${RUN_TAG}/checkpoints}"
    run_name="${RUN_NAME_OVERRIDE:-${exp_name_with_models}_${thinking_tag}_${maxlen_tag}_${terminal_tag}_${RUN_TAG}}"
  fi
  mkdir -p "${output_dir}"

  "${PYTHON_BIN}" -m opd_dynamics.train \
    --model_name_or_path "${STUDENT_MODEL}" \
    --teacher_model_name_or_path "${TEACHER_MODEL}" \
    --dataset_name "${TRAIN_DATA}" \
    --dataset_train_split train \
    --bf16 "${BF16}" \
    --dtype "${STUDENT_DTYPE}" \
    --teacher_dtype "${TEACHER_DTYPE}" \
    --teacher_load_in_4bit "${TEACHER_LOAD_IN_4BIT}" \
    --teacher_load_in_8bit "${TEACHER_LOAD_IN_8BIT}" \
    --teacher_bnb_4bit_quant_type "${TEACHER_BNB_4BIT_QUANT_TYPE}" \
    --teacher_bnb_4bit_compute_dtype "${TEACHER_BNB_4BIT_COMPUTE_DTYPE}" \
    --teacher_use_bnb_nested_quant "${TEACHER_USE_BNB_NESTED_QUANT}" \
    --teacher_bnb_8bit_threshold "${TEACHER_BNB_8BIT_THRESHOLD}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --max_length "${MAX_LENGTH}" \
    --trajectory_source student \
    --reward_mode "${reward_mode}" \
    --log_reward_clip_min "${LOG_REWARD_CLIP_MIN}" \
    --log_reward_clip_max "${LOG_REWARD_CLIP_MAX}" \
    --log_reward_tanh_temperature "${LOG_REWARD_TANH_TEMPERATURE}" \
    --power_reward_alpha "${POWER_REWARD_ALPHA}" \
    --token_loss_normalization "${TOKEN_LOSS_NORMALIZATION}" \
    --reward_normalization "${REWARD_NORMALIZATION}" \
    --terminal_stop_target "${TERMINAL_STOP_TARGET}" \
    --reward_position_plot_steps "${REWARD_POSITION_PLOT_STEPS}" \
    --reward_position_max_index "${REWARD_POSITION_MAX_INDEX}" \
    --reward_position_smooth_window "${REWARD_POSITION_SMOOTH_WINDOW}" \
    --beta 1.0 \
    --kl_mix_mode "${KL_MIX_MODE}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to "${REPORT_TO}" \
    --seed "${SEED}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --trust_remote_code "${TRUST_REMOTE_CODE}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_only_model "${SAVE_ONLY_MODEL}" \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --use_vllm "${USE_VLLM}" \
    --vllm_mode "${VLLM_MODE}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_test_names math \
    --eval_test_paths "${EVAL_DATA}" \
    --eval_test_samples "${EVAL_TEST_SAMPLES}" \
    --eval_temperature "${EVAL_TEMPERATURE}" \
    --eval_n "${EVAL_N}" \
    --eval_k "${EVAL_K}" \
    --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
    --max_steps "${MAX_STEPS}" \
    --enable_thinking "${ENABLE_THINKING}"
}
