from dataclasses import dataclass, field

from trl.experimental.gold.gold_config import GOLDConfig


@dataclass
class FactorizedGOLDConfig(GOLDConfig):
    trajectory_source: str = field(
        default="student",
        metadata={"help": "Trajectory source: one of `teacher`, `student`, or `dataset`."},
    )
    eval_test_names: str = field(
        default="",
        metadata={"help": "Comma-separated evaluation dataset names."},
    )
    eval_test_paths: str = field(
        default="",
        metadata={"help": "Comma-separated JSON/JSONL paths for evaluation datasets."},
    )
    eval_test_samples: str = field(
        default="all",
        metadata={"help": "Sample limits per evaluation dataset. Single value or comma-separated list."},
    )
    eval_temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for evaluation generations."},
    )
    eval_n: int = field(
        default=1,
        metadata={"help": "How many responses to sample per prompt during evaluation."},
    )
    eval_k: int = field(
        default=1,
        metadata={
            "help": "How many of the sampled responses are used for metrics. The evaluator reports acc=avg@k and pass=pass@k, and requires eval_n >= eval_k."
        },
    )
    eval_max_new_tokens: int | None = field(
        default=None,
        metadata={"help": "Max new tokens during eval. Falls back to max_completion_length when unset."},
    )
    enable_thinking: bool | None = field(
        default=False,
        metadata={"help": "Whether chat template rendering should enable thinking mode."},
    )
    teacher_dtype: str | None = field(
        default=None,
        metadata={
            "help": "Optional explicit dtype for teacher model loading, e.g. bfloat16 or float16. When unset, falls back to --dtype.",
        },
    )
    kl_mix_mode: str = field(
        default="jsd",
        metadata={
            "help": "Soft-target loss for intermediate beta values: `jsd` keeps the original generalized JSD; `linear_kl` uses beta * reverse-KL + (1 - beta) * forward-KL.",
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.trajectory_source not in {"teacher", "student", "dataset"}:
            raise ValueError(
                f"trajectory_source must be one of teacher/student/dataset, got {self.trajectory_source!r}."
            )
        if self.eval_n < 1:
            raise ValueError(f"eval_n must be >= 1, got {self.eval_n}.")
        if self.eval_k < 1:
            raise ValueError(f"eval_k must be >= 1, got {self.eval_k}.")
        if self.eval_k > self.eval_n:
            raise ValueError(f"eval_k ({self.eval_k}) must be <= eval_n ({self.eval_n}).")
        if self.kl_mix_mode not in {"jsd", "linear_kl"}:
            raise ValueError(f"kl_mix_mode must be one of jsd/linear_kl, got {self.kl_mix_mode!r}.")

