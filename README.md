<div align="center">

# PowerOPD

**Stabilizing On-Policy Distillation with Bounded Power Transformation**

[![arXiv](https://img.shields.io/badge/arXiv-2606.17199-b31b1b.svg)](https://arxiv.org/abs/2606.17199)
[![GitHub](https://img.shields.io/badge/GitHub-EIT--NLP%2FPowerOPD-181717?logo=github)](https://github.com/EIT-NLP/PowerOPD)

</div>

---

## 🎉 News

- **[2025-06]** PowerOPD paper released on arXiv: [arXiv:2606.17199](https://arxiv.org/abs/2606.17199).

---

## 📖 Overview

On-policy distillation (OPD) trains a student model to imitate a teacher on the student's own generated trajectories. The standard log-ratio reward `log p_T(o_t) − log p_S(o_t)` is unbounded: it can be arbitrarily large when the student assigns near-zero probability to a token, causing training instability and high memory cost.

**PowerOPD** replaces the log-ratio reward with a **power-transformed** version:

$$r_t^{(\alpha)} = p_T(o_t)^\alpha - p_S(o_t)^\alpha$$

The power transformation has three key properties:
- **Bounded**: rewards are clipped to `[−1, 1]` by construction, eliminating the instability from extreme log-ratios.
- **Sign-consistent**: the reward is positive when the teacher assigns higher probability than the student, and negative otherwise — same direction as log-ratio OPD.
- **Recovers log-ratio in the limit**: as α → 0, `r_t^(α)` converges (after rescaling) to the standard log-ratio reward.

Compared to vanilla OPD (log-ratio reward), PowerOPD achieves:

| Metric | Improvement |
|---|---|
| Pass@8 (6 math benchmarks) | **+5.71** |
| Avg@8 (6 math benchmarks) | **+6.37** |
| Wall-clock training time | **−59.2%** |
| Peak GPU memory | **−23.1%** |

---

## ✨ Getting Started

### Environment

```bash
git clone https://github.com/EIT-NLP/PowerOPD
cd PowerOPD
pip install -r requirements.txt
```

The training code uses a local TRL checkout included in `trl/`. Set up PYTHONPATH automatically via the launch scripts.

### Model Setup

Download your student and teacher models and place them (or symlink them) under `models/`:

```
models/
  Qwen3-1.7B-Base/
  Qwen3-4B/
```

The scripts default to `${PROJECT_ROOT}/models/Qwen3-1.7B-Base` and `${PROJECT_ROOT}/models/Qwen3-4B`. Override with environment variables:

```bash
export STUDENT_MODEL=/path/to/student
export TEACHER_MODEL=/path/to/teacher
```

### Training

**PowerOPD** (recommended):

```bash
bash scripts/run_power_diff.sh
```

Key hyperparameters (all overridable via environment variables):

| Variable | Default | Description |
|---|---|---|
| `POWER_REWARD_ALPHA` | `0.4` | Power exponent α (larger = stronger compression) |
| `TOKEN_LOSS_NORMALIZATION` | `window_token_mean` | Loss normalization across tokens |
| `TERMINAL_STOP_TARGET` | `mask` | How to handle the final stop token |
| `MAX_STEPS` | `1500` | Total training steps |
| `LEARNING_RATE` | `5e-7` | Learning rate |
| `GRADIENT_ACCUMULATION_STEPS` | `32` | Gradient accumulation steps |
| `EVAL_STEPS` | `30` | Evaluation frequency |

**Example with custom settings:**

```bash
POWER_REWARD_ALPHA=1.0 \
STUDENT_MODEL=/path/to/student \
TEACHER_MODEL=/path/to/teacher \
bash scripts/run_power_diff.sh
```

**Teacher quantization** (reduces GPU memory for large teachers):

```bash
TEACHER_QUANT_MODE=4bit bash scripts/run_power_diff_teacher_quant.sh
```

### Baseline Comparisons

Standard log-ratio OPD and other reward variants are provided for comparison:

```bash
bash scripts/run_log_ratio.sh   # vanilla OPD (log-ratio reward)
bash scripts/run_prob_diff.sh   # probability-difference reward
bash scripts/run_sqrt_diff.sh   # square-root power reward (α = 0.5)
```

### Experiment Tracking

The scripts report to [SwanLab](https://swanlab.cn) by default. Set your API key:

```bash
export SWANLAB_API_KEY=your_api_key
```

To disable tracking:

```bash
REPORT_TO=none bash scripts/run_power_diff.sh
```

---

## 📄 Data

Training and evaluation data are included under `data/`:

- `data/train/deepscaler_conversation.jsonl` — DeepScaler math reasoning training set
- `data/eval/math_test.jsonl` — MATH-500 evaluation set

---

## 📨 Contact

For questions about the paper or codebase, open an issue or contact the authors via the arXiv paper page.

---

## 🎈 Citation

```bibtex
@article{zhao2025poweropd,
  title={PowerOPD: Stabilizing On-Policy Distillation with Bounded Power Transformation},
  author={Zhao, Anhao and Tong, Junlong and Fan, Yingqi and Nie, Ping and Li, Wenjie and Shen, Xiaoyu},
  journal={arXiv preprint arXiv:2606.17199},
  year={2025}
}
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EIT-NLP/PowerOPD&type=Date)](https://star-history.com/#EIT-NLP/PowerOPD&Date)
