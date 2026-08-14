from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback


SIGNAL_NAMES = (
    "reward",
    "student_entropy",
    "teacher_entropy",
    "student_top1_prob",
    "teacher_top1_prob",
    "teacher_top1_gap",
)
DIVERGING_SIGNALS = {"reward", "teacher_top1_gap"}
SIGNAL_LABELS = {
    "reward": "Mean reward",
    "student_entropy": "Student entropy",
    "teacher_entropy": "Teacher entropy",
    "student_top1_prob": "Student top-1 prob",
    "teacher_top1_prob": "Teacher top-1 prob",
    "teacher_top1_gap": "Teacher top-1 prob - student prob on teacher top-1",
}


def build_reward_position_signals(
    reward: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        student_log_probs = student_log_probs.detach().float()
        teacher_log_probs = teacher_log_probs.detach().float()

        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()
        teacher_top1_logp, teacher_top1_ids = teacher_log_probs.max(dim=-1)
        student_prob_on_teacher_top1 = student_probs.gather(
            -1,
            teacher_top1_ids.unsqueeze(-1),
        ).squeeze(-1)

        return {
            "reward": reward.detach().float(),
            "student_entropy": -(student_probs * student_log_probs).sum(dim=-1),
            "teacher_entropy": -(teacher_probs * teacher_log_probs).sum(dim=-1),
            "student_top1_prob": student_log_probs.max(dim=-1).values.exp(),
            "teacher_top1_prob": teacher_top1_logp.exp(),
            "teacher_top1_gap": teacher_top1_logp.exp() - student_prob_on_teacher_top1,
        }


class RewardPositionTracker:
    """Token-level signal aggregation by completion position."""

    def __init__(self, output_dir: str, plot_steps: int, max_index: int, smooth_window: int):
        self.plot_steps = int(plot_steps)
        self.max_index = int(max_index)
        self.smooth_window = int(smooth_window)
        self.enabled = self.plot_steps > 0 and self.max_index > 0
        self.output_dir = Path(output_dir) / "reward_position"
        self.signal_sums = {
            name: np.zeros(self.max_index, dtype=np.float64) for name in SIGNAL_NAMES
        }
        self.count = np.zeros(self.max_index, dtype=np.float64)
        self.last_saved_step = -1

    @classmethod
    def from_args(cls, args) -> "RewardPositionTracker":
        max_index = int(args.reward_position_max_index or args.max_completion_length or 0)
        return cls(
            output_dir=args.output_dir,
            plot_steps=args.reward_position_plot_steps,
            max_index=max_index,
            smooth_window=args.reward_position_smooth_window,
        )

    def record(self, signals: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        if not self.enabled:
            return

        missing = [name for name in SIGNAL_NAMES if name not in signals]
        if missing:
            raise ValueError(f"Missing reward position signals: {missing}")

        width = min([int(mask.shape[1]), self.max_index] + [int(signals[name].shape[1]) for name in SIGNAL_NAMES])
        if width <= 0:
            return

        with torch.no_grad():
            mask_slice = mask[:, :width].detach().bool()
            signal_slices = {}
            for name in SIGNAL_NAMES:
                values = signals[name][:, :width].detach().float()
                signal_slices[name] = values
                mask_slice = mask_slice & torch.isfinite(values)

            if not mask_slice.any():
                return

            count = mask_slice.sum(dim=0).cpu().numpy()
            signal_sums = {
                name: values.masked_fill(~mask_slice, 0.0).sum(dim=0).cpu().numpy()
                for name, values in signal_slices.items()
            }

        self.count[:width] += count
        for name in SIGNAL_NAMES:
            self.signal_sums[name][:width] += signal_sums[name]

    def maybe_save(self, step: int, should_write: bool = True) -> None:
        if not self.enabled or step <= 0 or step == self.last_saved_step:
            return
        should_save_initial = self.last_saved_step < 0 and self.count.sum() > 0
        if not should_save_initial and step % self.plot_steps != 0:
            return

        if should_write and self.count.sum() > 0:
            self._save(step)
        self.reset_window()
        self.last_saved_step = step

    def reset_window(self) -> None:
        if not self.enabled:
            return
        self.count.fill(0.0)
        for values in self.signal_sums.values():
            values.fill(0.0)

    def _smooth_sum_count(self, signal_sum: np.ndarray, count: np.ndarray) -> np.ndarray:
        smooth = np.full(self.max_index, np.nan, dtype=np.float64)
        valid = count > 0
        if not valid.any():
            return smooth
        if self.smooth_window <= 1:
            smooth[valid] = signal_sum[valid] / count[valid]
            return smooth

        window = min(self.smooth_window, self.max_index)
        kernel = np.ones(window, dtype=np.float64)
        smooth_sum = np.convolve(signal_sum, kernel, mode="same")
        smooth_count = np.convolve(count, kernel, mode="same")
        smooth_valid = smooth_count > 0
        smooth[smooth_valid] = smooth_sum[smooth_valid] / smooth_count[smooth_valid]
        return smooth

    def _window_values(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        valid = self.count > 0
        values = {}
        for name in SIGNAL_NAMES:
            mean = np.full(self.max_index, np.nan, dtype=np.float64)
            mean[valid] = self.signal_sums[name][valid] / self.count[valid]
            smooth = self._smooth_sum_count(self.signal_sums[name], self.count)
            values[name] = (mean, smooth)
        return values

    def _save(self, step: int) -> None:
        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        values = self._window_values()
        self._write_step_csv(step_dir, values)
        for name, (_mean, smooth) in values.items():
            self._plot_window(step, name, smooth, step_dir)
        self._rebuild_heatmaps_from_disk()

    def _write_step_csv(self, step_dir: Path, values: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        csv_path = step_dir / "metrics.csv"
        header = ["token_index", "count"]
        for name in SIGNAL_NAMES:
            header.extend([f"{name}_mean", f"{name}_smooth"])

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for token_index in range(self.max_index):
                if self.count[token_index] <= 0:
                    continue
                row = [token_index, int(self.count[token_index])]
                for name in SIGNAL_NAMES:
                    mean, smooth = values[name]
                    row.extend([float(mean[token_index]), float(smooth[token_index])])
                writer.writerow(row)

    def _load_step_records(self) -> list[dict[str, object]]:
        records = []
        if not self.output_dir.exists():
            return records

        for step_dir in sorted(self.output_dir.glob("step_*")):
            metrics_path = step_dir / "metrics.csv"
            if not metrics_path.exists():
                continue
            try:
                step = int(step_dir.name.removeprefix("step_"))
            except ValueError:
                continue

            matrices = {name: np.full(self.max_index, np.nan, dtype=np.float64) for name in SIGNAL_NAMES}
            counts = np.zeros(self.max_index, dtype=np.float64)
            with metrics_path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    token_index = int(row["token_index"])
                    if token_index < 0 or token_index >= self.max_index:
                        continue
                    counts[token_index] = float(row["count"])
                    for name in SIGNAL_NAMES:
                        matrices[name][token_index] = float(row[f"{name}_smooth"])

            observed = np.nonzero(counts > 0)[0]
            records.append(
                {
                    "step": step,
                    "step_dir": step_dir,
                    "signals": matrices,
                    "total_count": int(counts.sum()),
                    "max_observed_index": int(observed.max()) if observed.size else -1,
                }
            )

        records.sort(key=lambda record: int(record["step"]))
        return records

    def _write_summary(self, records: list[dict[str, object]]) -> None:
        summary_path = self.output_dir / "summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "step_dir", "total_count", "max_observed_index"])
            for record in records:
                step_dir = record["step_dir"]
                assert isinstance(step_dir, Path)
                writer.writerow(
                    [
                        int(record["step"]),
                        step_dir.name,
                        int(record["total_count"]),
                        int(record["max_observed_index"]),
                    ]
                )

    def _rebuild_heatmaps_from_disk(self) -> None:
        records = self._load_step_records()
        self._write_summary(records)
        if not records:
            return

        heatmap_dir = self.output_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for name in SIGNAL_NAMES:
            matrix = np.stack([record["signals"][name] for record in records], axis=0)
            steps = [int(record["step"]) for record in records]
            self._plot_heatmap(name, matrix, steps, heatmap_dir)

    def _import_pyplot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _plot_window(self, step: int, name: str, smooth: np.ndarray, step_dir: Path) -> None:
        plt = self._import_pyplot()
        x = np.arange(self.max_index)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x, smooth, linewidth=1.5)
        if name in DIVERGING_SIGNALS:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, max(1, self.max_index - 1))
        ax.set_xlabel("Completion token index")
        ax.set_ylabel(SIGNAL_LABELS[name])
        ax.set_title(f"{SIGNAL_LABELS[name]} by token index, step {step}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(step_dir / f"{name}.png", dpi=160)
        plt.close(fig)

    def _plot_heatmap(self, name: str, matrix: np.ndarray, steps: list[int], heatmap_dir: Path) -> None:
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            return

        plt = self._import_pyplot()
        norm = None
        cmap = "viridis"
        if name in DIVERGING_SIGNALS:
            from matplotlib.colors import TwoSlopeNorm

            limit = float(max(abs(np.nanpercentile(finite, 5)), abs(np.nanpercentile(finite, 95)), 1e-12))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            cmap = "coolwarm"

        height = max(3.0, min(8.0, 0.3 * len(steps) + 2.0))
        fig, ax = plt.subplots(figsize=(10, height))
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xlabel("Completion token index")
        ax.set_ylabel("Saved step")
        tick_count = min(8, len(steps))
        tick_positions = np.linspace(0, len(steps) - 1, tick_count, dtype=int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([str(steps[idx]) for idx in tick_positions])
        fig.colorbar(image, ax=ax, label=SIGNAL_LABELS[name])
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{name}.png", dpi=160)
        plt.close(fig)


class RewardPositionCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        tracker = self.trainer.reward_position_tracker
        should_write = self.trainer.accelerator.is_main_process
        tracker.maybe_save(int(state.global_step), should_write=should_write)
