#!/usr/bin/env bash
set -euo pipefail

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-reward_diff_opd}"
export REPORT_TO="${REPORT_TO:-swanlab}"
# TEACHER_MODEL defaults to ${PROJECT_ROOT}/models/Qwen3-4B from common.sh; override here if needed
export TERMINAL_STOP_TARGET="${TERMINAL_STOP_TARGET:-mask}"
export REWARD_NORMALIZATION="${REWARD_NORMALIZATION:-sign_max_abs}"
export TOKEN_LOSS_NORMALIZATION="${TOKEN_LOSS_NORMALIZATION:-window_token_mean}"

source "$(dirname "$0")/common.sh"
run_opd_dynamics sampled_prob_diff prob_diff_reward
