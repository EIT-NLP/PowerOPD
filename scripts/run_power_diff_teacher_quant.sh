#!/usr/bin/env bash
set -euo pipefail

# Model paths
# STUDENT_MODEL and TEACHER_MODEL default to ${PROJECT_ROOT}/models/* from common.sh; override here if needed

# Teacher quantization: 4bit, 8bit, or none
export TEACHER_QUANT_MODE="${TEACHER_QUANT_MODE:-4bit}"
export TEACHER_BNB_4BIT_QUANT_TYPE="${TEACHER_BNB_4BIT_QUANT_TYPE:-nf4}"
export TEACHER_BNB_4BIT_COMPUTE_DTYPE="${TEACHER_BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
export TEACHER_USE_BNB_NESTED_QUANT="${TEACHER_USE_BNB_NESTED_QUANT:-False}"
export TEACHER_BNB_8BIT_THRESHOLD="${TEACHER_BNB_8BIT_THRESHOLD:-6.0}"

case "${TEACHER_QUANT_MODE,,}" in
  4bit|int4|w4a16)
    export TEACHER_LOAD_IN_4BIT=True
    export TEACHER_LOAD_IN_8BIT=False
    ;;
  8bit|int8|w8a16)
    export TEACHER_LOAD_IN_4BIT=False
    export TEACHER_LOAD_IN_8BIT=True
    ;;
  none|false|off)
    export TEACHER_LOAD_IN_4BIT=False
    export TEACHER_LOAD_IN_8BIT=False
    ;;
  *)
    echo "Unsupported TEACHER_QUANT_MODE=${TEACHER_QUANT_MODE}. Use 4bit, 8bit, or none." >&2
    exit 2
    ;;
esac

# Logging / reporting
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-power_diff_opd_teacher_quant}"
export REPORT_TO="${REPORT_TO:-swanlab}"

# OPD reward and terminal handling
export TERMINAL_STOP_TARGET="${TERMINAL_STOP_TARGET:-mask}"
export TOKEN_LOSS_NORMALIZATION="${TOKEN_LOSS_NORMALIZATION:-window_token_mean}"
export REWARD_NORMALIZATION="${REWARD_NORMALIZATION:-none}"
export POWER_REWARD_ALPHA="${POWER_REWARD_ALPHA:-5}"

# Training / generation knobs
export BF16="${BF16:-True}"
export STUDENT_DTYPE="${STUDENT_DTYPE:-float32}"
export TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-1500}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
export MAX_LENGTH="${MAX_LENGTH:-6144}"
export USE_VLLM="${USE_VLLM:-True}"
export VLLM_MODE="${VLLM_MODE:-colocate}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export ENABLE_THINKING="${ENABLE_THINKING:-False}"

# Eval / save knobs
export EVAL_STEPS="${EVAL_STEPS:-30}"
export EVAL_TEST_SAMPLES="${EVAL_TEST_SAMPLES:-100}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
export EVAL_N="${EVAL_N:-1}"
export EVAL_K="${EVAL_K:-1}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"

# Misc
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export SEED="${SEED:-42}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
export KL_MIX_MODE="${KL_MIX_MODE:-jsd}"
export REWARD_POSITION_PLOT_STEPS="${REWARD_POSITION_PLOT_STEPS:-10}"
export REWARD_POSITION_MAX_INDEX="${REWARD_POSITION_MAX_INDEX:-${MAX_COMPLETION_LENGTH}}"
export REWARD_POSITION_SMOOTH_WINDOW="${REWARD_POSITION_SMOOTH_WINDOW:-16}"

source "$(dirname "$0")/common.sh"
run_opd_dynamics sampled_power_diff power_diff_reward

