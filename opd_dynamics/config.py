from __future__ import annotations

from dataclasses import dataclass, field

from factorized_gold.config import FactorizedGOLDConfig

from .terminal_stop import VALID_TERMINAL_STOP_TARGETS


@dataclass
class OPDDynamicsConfig(FactorizedGOLDConfig):
    reward_mode: str = field(
        default="sampled_log_ratio",
        metadata={
            "help": (
                "Sampled-token reward: `sampled_log_ratio`, `sampled_log_clip`, `sampled_log_tanh`, "
                "`sampled_prob_diff`, `sampled_sqrt_diff`, or `sampled_power_diff`."
            )
        },
    )
    log_reward_clip_min: float = field(
        default=-5.0,
        metadata={"help": "Lower bound for sampled_log_clip reward."},
    )
    log_reward_clip_max: float = field(
        default=5.0,
        metadata={"help": "Upper bound for sampled_log_clip reward."},
    )
    log_reward_tanh_temperature: float = field(
        default=5.0,
        metadata={"help": "Temperature divisor for sampled_log_tanh reward."},
    )
    power_reward_alpha: float = field(
        default=0.1,
        metadata={"help": "Alpha for sampled_power_diff reward: p_T(o_t)^alpha - p_S(o_t)^alpha."},
    )
    token_loss_normalization: str = field(
        default="microbatch_mean",
        metadata={"help": "Token loss denominator: `microbatch_mean` or `window_token_mean`."},
    )
    reward_normalization: str = field(
        default="none",
        metadata={"help": "Reward normalization over the accumulation window: `none`, `sign_max_abs`, or `zscore`."},
    )
    terminal_stop_target: str = field(
        default="im_end",
        metadata={"help": "Terminal stop handling: `raw`, `mask`, `im_end`, or `cross_eos`."},
    )
    reward_position_plot_steps: int = field(
        default=0,
        metadata={"help": "Save reward-by-token-index plots every N optimizer steps. 0 disables."},
    )
    reward_position_max_index: int = field(
        default=0,
        metadata={"help": "Maximum completion token index count to plot. 0 uses max_completion_length."},
    )
    reward_position_smooth_window: int = field(
        default=16,
        metadata={"help": "Token-index smoothing window for reward position plots."},
    )
    teacher_load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 4-bit weights for scoring-only forward passes."},
    )
    teacher_load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 8-bit weights for scoring-only forward passes."},
    )
    teacher_bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "bitsandbytes 4-bit quantization type for the teacher model, e.g. `nf4` or `fp4`."},
    )
    teacher_bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for teacher bitsandbytes 4-bit layers."},
    )
    teacher_use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use bitsandbytes nested/double quantization for the teacher model."},
    )
    teacher_bnb_8bit_threshold: float = field(
        default=6.0,
        metadata={"help": "bitsandbytes llm_int8_threshold for teacher 8-bit loading."},
    )

    def __post_init__(self):
        super().__post_init__()
        valid = {
            "sampled_log_ratio",
            "sampled_log_clip",
            "sampled_log_tanh",
            "sampled_prob_diff",
            "sampled_sqrt_diff",
            "sampled_power_diff",
        }
        if self.reward_mode not in valid:
            raise ValueError(f"reward_mode must be one of {sorted(valid)}, got {self.reward_mode!r}.")
        if self.log_reward_clip_min >= self.log_reward_clip_max:
            raise ValueError("log_reward_clip_min must be smaller than log_reward_clip_max.")
        if self.log_reward_tanh_temperature <= 0:
            raise ValueError("log_reward_tanh_temperature must be > 0.")
        if self.power_reward_alpha <= 0:
            raise ValueError("power_reward_alpha must be > 0.")
        valid_token_loss_normalization = {"microbatch_mean", "window_token_mean"}
        if self.token_loss_normalization not in valid_token_loss_normalization:
            raise ValueError(
                f"token_loss_normalization must be one of {sorted(valid_token_loss_normalization)}, "
                f"got {self.token_loss_normalization!r}."
            )
        valid_reward_normalization = {"none", "sign_max_abs", "zscore"}
        if self.reward_normalization not in valid_reward_normalization:
            raise ValueError(
                f"reward_normalization must be one of {sorted(valid_reward_normalization)}, "
                f"got {self.reward_normalization!r}."
            )
        if self.terminal_stop_target not in VALID_TERMINAL_STOP_TARGETS:
            raise ValueError(
                f"terminal_stop_target must be one of {sorted(VALID_TERMINAL_STOP_TARGETS)}, "
                f"got {self.terminal_stop_target!r}."
            )
        if self.reward_position_plot_steps < 0:
            raise ValueError("reward_position_plot_steps must be >= 0.")
        if self.reward_position_max_index < 0:
            raise ValueError("reward_position_max_index must be >= 0.")
        if self.reward_position_smooth_window < 1:
            raise ValueError("reward_position_smooth_window must be >= 1.")
        if self.teacher_load_in_4bit and self.teacher_load_in_8bit:
            raise ValueError("Only one of teacher_load_in_4bit and teacher_load_in_8bit can be enabled.")


@dataclass
class REOPOLDNoRhoConfig(FactorizedGOLDConfig):
    reopold_lambda: float = field(
        default=0.3,
        metadata={"help": "REOPOLD clipping coefficient lambda. Paper default: 0.3."},
    )
    reopold_entropy_top_fraction: float = field(
        default=0.2,
        metadata={"help": "Keep top beta fraction of token entropies in phase II. Paper default beta: 0.2."},
    )
    reopold_switch_step: int = field(
        default=500,
        metadata={"help": "Switch from reward filtering to entropy filtering."},
    )
    terminal_stop_target: str = field(
        default="im_end",
        metadata={"help": "Terminal stop handling: `raw`, `mask`, `im_end`, or `cross_eos`."},
    )
    reward_position_plot_steps: int = field(
        default=0,
        metadata={"help": "Save reward-by-token-index plots every N optimizer steps. 0 disables."},
    )
    reward_position_max_index: int = field(
        default=0,
        metadata={"help": "Maximum completion token index count to plot. 0 uses max_completion_length."},
    )
    reward_position_smooth_window: int = field(
        default=16,
        metadata={"help": "Token-index smoothing window for reward position plots."},
    )
    teacher_load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 4-bit weights for scoring-only forward passes."},
    )
    teacher_load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the teacher model with bitsandbytes 8-bit weights for scoring-only forward passes."},
    )
    teacher_bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "bitsandbytes 4-bit quantization type for the teacher model, e.g. `nf4` or `fp4`."},
    )
    teacher_bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for teacher bitsandbytes 4-bit layers."},
    )
    teacher_use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use bitsandbytes nested/double quantization for the teacher model."},
    )
    teacher_bnb_8bit_threshold: float = field(
        default=6.0,
        metadata={"help": "bitsandbytes llm_int8_threshold for teacher 8-bit loading."},
    )

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.reopold_lambda < 1.0):
            raise ValueError("reopold_lambda must be in (0, 1).")
        if not (0.0 < self.reopold_entropy_top_fraction <= 1.0):
            raise ValueError("reopold_entropy_top_fraction must be in (0, 1].")
        if self.reopold_switch_step < 0:
            raise ValueError("reopold_switch_step must be >= 0.")
        if self.terminal_stop_target not in VALID_TERMINAL_STOP_TARGETS:
            raise ValueError(
                f"terminal_stop_target must be one of {sorted(VALID_TERMINAL_STOP_TARGETS)}, "
                f"got {self.terminal_stop_target!r}."
            )
        if self.reward_position_plot_steps < 0:
            raise ValueError("reward_position_plot_steps must be >= 0.")
        if self.reward_position_max_index < 0:
            raise ValueError("reward_position_max_index must be >= 0.")
        if self.reward_position_smooth_window < 1:
            raise ValueError("reward_position_smooth_window must be >= 1.")
        if self.teacher_load_in_4bit and self.teacher_load_in_8bit:
            raise ValueError("Only one of teacher_load_in_4bit and teacher_load_in_8bit can be enabled.")
