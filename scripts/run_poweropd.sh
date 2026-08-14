#!/usr/bin/env bash
set -euo pipefail

export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-power_diff_opd}"
export REPORT_TO="${REPORT_TO:-swanlab}"

export TERMINAL_STOP_TARGET="${TERMINAL_STOP_TARGET:-mask}"
export TOKEN_LOSS_NORMALIZATION="${TOKEN_LOSS_NORMALIZATION:-window_token_mean}"
export POWER_REWARD_ALPHA="${POWER_REWARD_ALPHA:-5}"

source "$(dirname "$0")/common.sh"
run_opd_dynamics sampled_power_diff power_diff_reward
