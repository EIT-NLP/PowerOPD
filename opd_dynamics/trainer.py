from __future__ import annotations

from typing import Any

import torch

from factorized_gold.stop_tokens import apply_dual_eos_stop_token_ids
from factorized_gold.trainer import FactorizedGOLDTrainer
from trl.experimental.utils import empty_cache

from .reward_position import RewardPositionCallback, RewardPositionTracker, build_reward_position_signals
from .terminal_stop import apply_cross_eos_terminal_stop_target, apply_terminal_stop_target


class OPDDynamicsTrainer(FactorizedGOLDTrainer):
    """Factorized GOLD trainer with sampled-token reward objectives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_mode = getattr(self.args, "reward_mode", "sampled_log_ratio")
        self.log_reward_clip_min = float(getattr(self.args, "log_reward_clip_min", -5.0))
        self.log_reward_clip_max = float(getattr(self.args, "log_reward_clip_max", 5.0))
        self.log_reward_tanh_temperature = float(getattr(self.args, "log_reward_tanh_temperature", 5.0))
        self.power_reward_alpha = float(getattr(self.args, "power_reward_alpha", 0.1))
        self.token_loss_normalization = getattr(self.args, "token_loss_normalization", "microbatch_mean")
        self.reward_normalization = getattr(self.args, "reward_normalization", "none")
        self.terminal_stop_target = getattr(self.args, "terminal_stop_target", "im_end")
        apply_dual_eos_stop_token_ids(self)
        self.reward_position_tracker = RewardPositionTracker.from_args(self.args)
        if self.reward_position_tracker.enabled:
            self.add_callback(RewardPositionCallback(self))

    def _uses_window_reward_cache(self) -> bool:
        return self.token_loss_normalization == "window_token_mean" or self.reward_normalization != "none"

    def _record_train_scalar(self, key: str, value: torch.Tensor | float | int) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().cpu().item())
        self._metrics["train"][key].append(float(value))

    def _record_terminal_stop_stats(self, stats: dict[str, int]) -> None:
        for key, value in stats.items():
            self._record_train_scalar(key, value)

    def _summarize_tensor(self, prefix: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().float()
        self._record_train_scalar(f"{prefix}_mean", values.mean())
        self._record_train_scalar(f"{prefix}_std", values.std(unbiased=False))
        self._record_train_scalar(f"{prefix}_min", values.min())
        self._record_train_scalar(f"{prefix}_max", values.max())
        self._record_train_scalar(f"{prefix}_p05", torch.quantile(values, 0.05))
        self._record_train_scalar(f"{prefix}_p95", torch.quantile(values, 0.95))

    def _select_reward_full(
        self,
        log_ratio_full: torch.Tensor,
        log_clip_full: torch.Tensor,
        log_tanh_full: torch.Tensor,
        prob_diff_full: torch.Tensor,
        sqrt_diff_full: torch.Tensor,
        power_diff_full: torch.Tensor,
    ) -> torch.Tensor:
        if self.reward_mode == "sampled_log_ratio":
            return log_ratio_full
        if self.reward_mode == "sampled_log_clip":
            return log_clip_full
        if self.reward_mode == "sampled_log_tanh":
            return log_tanh_full
        if self.reward_mode == "sampled_prob_diff":
            return prob_diff_full
        if self.reward_mode == "sampled_sqrt_diff":
            return sqrt_diff_full
        if self.reward_mode == "sampled_power_diff":
            return power_diff_full
        raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")

    def _build_reward_components(
        self,
        inputs: dict[str, torch.Tensor | Any],
        shifted_student_logits: torch.Tensor,
        shifted_teacher_logits: torch.Tensor,
    ) -> dict[str, Any]:
        prompt_length = inputs["prompts"].shape[1]
        shifted_labels = inputs["labels"][:, prompt_length:]
        mask = shifted_labels != -100
        if self.terminal_stop_target == "cross_eos":
            terminal = apply_cross_eos_terminal_stop_target(shifted_labels, mask, self.processing_class)
            student_labels = terminal.student_labels
            teacher_labels = terminal.teacher_labels
            loss_labels = terminal.loss_labels
            effective_mask = terminal.mask
        else:
            terminal = apply_terminal_stop_target(
                shifted_labels,
                mask,
                self.processing_class,
                self.terminal_stop_target,
            )
            student_labels = terminal.labels
            teacher_labels = terminal.labels
            loss_labels = terminal.labels
            effective_mask = terminal.mask

        safe_student_labels = student_labels.masked_fill(~effective_mask, 0).long()
        safe_teacher_labels = teacher_labels.masked_fill(~effective_mask, 0).long()
        safe_loss_labels = loss_labels.masked_fill(~effective_mask, 0).long()
        student_log_probs = torch.log_softmax(shifted_student_logits.float(), dim=-1)
        teacher_log_probs = torch.log_softmax(shifted_teacher_logits.float(), dim=-1)

        student_logp_full = student_log_probs.gather(-1, safe_student_labels.unsqueeze(-1)).squeeze(-1)
        teacher_logp_full = teacher_log_probs.gather(-1, safe_teacher_labels.unsqueeze(-1)).squeeze(-1)
        loss_logp_full = student_log_probs.gather(-1, safe_loss_labels.unsqueeze(-1)).squeeze(-1)

        log_ratio_full = teacher_logp_full - student_logp_full
        log_clip_full = log_ratio_full.clamp(self.log_reward_clip_min, self.log_reward_clip_max)
        log_tanh_full = torch.tanh(log_ratio_full / self.log_reward_tanh_temperature)
        prob_diff_full = teacher_logp_full.exp() - student_logp_full.exp()
        sqrt_diff_full = teacher_logp_full.exp().sqrt() - student_logp_full.exp().sqrt()
        power_diff_full = torch.exp(self.power_reward_alpha * teacher_logp_full) - torch.exp(
            self.power_reward_alpha * student_logp_full
        )
        reward_full = self._select_reward_full(
            log_ratio_full,
            log_clip_full,
            log_tanh_full,
            prob_diff_full,
            sqrt_diff_full,
            power_diff_full,
        )
        token_entropy_full = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)

        return {
            "terminal_stats": terminal.stats,
            "effective_mask": effective_mask,
            "loss_labels": loss_labels,
            "loss_logp_full": loss_logp_full,
            "student_logp_full": student_logp_full,
            "teacher_logp_full": teacher_logp_full,
            "student_log_probs": student_log_probs,
            "teacher_log_probs": teacher_log_probs,
            "log_ratio_full": log_ratio_full,
            "log_clip_full": log_clip_full,
            "log_tanh_full": log_tanh_full,
            "prob_diff_full": prob_diff_full,
            "sqrt_diff_full": sqrt_diff_full,
            "power_diff_full": power_diff_full,
            "reward_full": reward_full,
            "token_entropy_full": token_entropy_full,
        }

    def _normalize_reward_full(
        self,
        reward_full: torch.Tensor,
        pos_scale: torch.Tensor,
        neg_scale: torch.Tensor,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
    ) -> torch.Tensor:
        if self.reward_normalization == "none":
            return reward_full
        eps = torch.tensor(1e-8, dtype=reward_full.dtype, device=reward_full.device)
        if self.reward_normalization == "zscore":
            norm_mean = norm_mean.to(device=reward_full.device, dtype=reward_full.dtype)
            norm_std = norm_std.to(device=reward_full.device, dtype=reward_full.dtype).clamp_min(eps)
            return (reward_full - norm_mean) / norm_std
        if self.reward_normalization != "sign_max_abs":
            raise ValueError(f"Unsupported reward_normalization: {self.reward_normalization}")
        pos_scale = pos_scale.to(device=reward_full.device, dtype=reward_full.dtype).clamp_min(eps)
        neg_scale = neg_scale.to(device=reward_full.device, dtype=reward_full.dtype).clamp_min(eps)
        return torch.where(
            reward_full > 0,
            reward_full / pos_scale,
            torch.where(reward_full < 0, reward_full / neg_scale, reward_full),
        )

    def _mode_code(self, value: str, mapping: dict[str, float]) -> float:
        return mapping.get(value, -1.0)

    def _fill_buffer(self, generation_batch: dict[str, torch.Tensor | Any], buffer_steps: int):
        super()._fill_buffer(generation_batch, buffer_steps)
        if self._uses_window_reward_cache():
            self._precompute_sampled_reward_window()

    def _empty_window_reward_records(self, records: list[tuple[int, dict[str, torch.Tensor | Any], torch.Tensor]]) -> None:
        for idx, inputs, mask in records:
            updated_inputs = dict(inputs)
            updated_inputs["sampled_window_reward"] = torch.zeros_like(mask, dtype=torch.float32)
            updated_inputs["sampled_window_mask"] = torch.zeros_like(mask, dtype=torch.bool)
            updated_inputs["sampled_window_loss_labels"] = torch.zeros_like(mask, dtype=torch.long)
            updated_inputs["sampled_window_token_count"] = torch.tensor(1.0, dtype=torch.float32, device=mask.device)
            self._buffered_inputs[idx] = updated_inputs

    def _precompute_sampled_reward_window(self) -> None:
        records = []
        flat_raw_reward = []
        flat_log_ratio = []
        flat_log_clip = []
        flat_log_tanh = []
        flat_prob_diff = []
        flat_sqrt_diff = []
        flat_power_diff = []
        flat_entropy = []
        flat_student_prob = []
        terminal_totals = {
            "terminal_stop_endoftext_count": 0,
            "terminal_stop_im_end_count": 0,
            "terminal_stop_missing_count": 0,
            "terminal_stop_canonicalized_count": 0,
            "terminal_stop_masked_count": 0,
            "terminal_stop_cross_eos_count": 0,
        }

        model_was_training = self.model.training
        self.model.eval()
        self.teacher_model.eval()
        try:
            with torch.no_grad():
                for idx, inputs in enumerate(getattr(self, "_buffered_inputs", [])):
                    if inputs is None or "labels" not in inputs:
                        continue
                    outputs_student = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                    )
                    outputs_teacher = self.teacher_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                    )
                    prompt_length = inputs["prompts"].shape[1]
                    components = self._build_reward_components(
                        inputs,
                        outputs_student.logits[:, prompt_length - 1 : -1, :],
                        outputs_teacher.logits[:, prompt_length - 1 : -1, :],
                    )
                    mask = components["effective_mask"].detach()
                    for key, value in components["terminal_stats"].items():
                        terminal_totals[key] += int(value)

                    reward_full = components["reward_full"].detach()
                    position_signals = None
                    if self.reward_position_tracker.enabled:
                        position_signals = build_reward_position_signals(
                            reward_full,
                            components["student_log_probs"],
                            components["teacher_log_probs"],
                        )
                        position_signals = {key: value.detach() for key, value in position_signals.items()}

                    records.append(
                        (
                            idx,
                            inputs,
                            mask,
                            components["loss_labels"].detach(),
                            reward_full,
                            position_signals,
                        )
                    )
                    if not mask.any():
                        continue
                    flat_raw_reward.append(reward_full[mask].detach())
                    flat_log_ratio.append(components["log_ratio_full"][mask].detach())
                    flat_log_clip.append(components["log_clip_full"][mask].detach())
                    flat_log_tanh.append(components["log_tanh_full"][mask].detach())
                    flat_prob_diff.append(components["prob_diff_full"][mask].detach())
                    flat_sqrt_diff.append(components["sqrt_diff_full"][mask].detach())
                    flat_power_diff.append(components["power_diff_full"][mask].detach())
                    flat_entropy.append(components["token_entropy_full"][mask].detach())
                    flat_student_prob.append(components["student_logp_full"][mask].exp().detach())
        finally:
            self.model.train(model_was_training)

        self._record_terminal_stop_stats(terminal_totals)
        self._record_train_scalar(
            "train_token_loss_normalization_mode",
            self._mode_code("window_token_mean" if self.token_loss_normalization == "window_token_mean" else "microbatch_mean", {"microbatch_mean": 0.0, "window_token_mean": 1.0}),
        )
        self._record_train_scalar(
            "train_reward_normalization_mode",
            self._mode_code(self.reward_normalization, {"none": 0.0, "sign_max_abs": 1.0, "zscore": 2.0}),
        )

        if not flat_raw_reward:
            self._empty_window_reward_records([(idx, inputs, mask) for idx, inputs, mask, *_ in records])
            self._record_train_scalar("train_window_token_count", 0)
            self._record_train_scalar("train_reward_pos_scale", 1.0)
            self._record_train_scalar("train_reward_neg_scale", 1.0)
            self._record_train_scalar("train_reward_norm_mean", 0.0)
            self._record_train_scalar("train_reward_norm_std", 1.0)
            empty_cache()
            return

        all_raw_reward = torch.cat(flat_raw_reward)
        all_log_ratio = torch.cat(flat_log_ratio)
        all_log_clip = torch.cat(flat_log_clip)
        all_log_tanh = torch.cat(flat_log_tanh)
        all_prob_diff = torch.cat(flat_prob_diff)
        all_sqrt_diff = torch.cat(flat_sqrt_diff)
        all_power_diff = torch.cat(flat_power_diff)
        all_entropy = torch.cat(flat_entropy)
        all_student_prob = torch.cat(flat_student_prob)
        window_token_count = int(all_raw_reward.numel())

        positive = all_raw_reward[all_raw_reward > 0]
        negative = all_raw_reward[all_raw_reward < 0]
        pos_scale = positive.max() if positive.numel() else torch.tensor(1.0, dtype=all_raw_reward.dtype, device=all_raw_reward.device)
        neg_scale = negative.abs().max() if negative.numel() else torch.tensor(1.0, dtype=all_raw_reward.dtype, device=all_raw_reward.device)
        norm_mean = all_raw_reward.mean()
        norm_std = all_raw_reward.std(unbiased=False)
        all_normalized_reward = self._normalize_reward_full(
            all_raw_reward,
            pos_scale,
            neg_scale,
            norm_mean,
            norm_std,
        ).detach()

        self._record_train_scalar("train_window_token_count", window_token_count)
        self._record_train_scalar("train_reward_pos_scale", pos_scale)
        self._record_train_scalar("train_reward_neg_scale", neg_scale)
        self._record_train_scalar("train_reward_norm_mean", norm_mean)
        self._record_train_scalar("train_reward_norm_std", norm_std)
        self._summarize_tensor("train_reward_raw", all_raw_reward)
        self._summarize_tensor("train_reward_normalized", all_normalized_reward)
        self._summarize_tensor("train_reward", all_normalized_reward)
        self._summarize_tensor("train_log_ratio", all_log_ratio)
        self._summarize_tensor("train_log_clip", all_log_clip)
        self._summarize_tensor("train_log_tanh", all_log_tanh)
        self._summarize_tensor("train_prob_diff", all_prob_diff)
        self._summarize_tensor("train_sqrt_diff", all_sqrt_diff)
        self._summarize_tensor("train_power_diff", all_power_diff)
        self._record_train_scalar("train_policy_entropy", all_entropy.mean())
        self._record_train_scalar("train_student_sampled_prob_mean", all_student_prob.mean())
        self._record_train_scalar("train_abs_prob_diff_gt_0.01", (all_prob_diff.abs() > 0.01).float().mean())
        self._record_train_scalar("train_abs_sqrt_diff_gt_0.01", (all_sqrt_diff.abs() > 0.01).float().mean())
        self._record_train_scalar("train_abs_power_diff_gt_0.01", (all_power_diff.abs() > 0.01).float().mean())
        self._record_train_scalar("train_abs_log_ratio_gt_0.1", (all_log_ratio.abs() > 0.1).float().mean())
        self._record_train_scalar("train_abs_log_clip_gt_0.1", (all_log_clip.abs() > 0.1).float().mean())
        self._record_train_scalar("train_abs_log_tanh_gt_0.1", (all_log_tanh.abs() > 0.1).float().mean())

        window_token_count_tensor = torch.tensor(float(max(window_token_count, 1)), dtype=torch.float32)
        for idx, inputs, mask, loss_labels, raw_reward_full, position_signals in records:
            normalized_reward_full = self._normalize_reward_full(raw_reward_full, pos_scale, neg_scale, norm_mean, norm_std).detach()
            updated_inputs = dict(inputs)
            updated_inputs["sampled_window_reward"] = normalized_reward_full
            updated_inputs["sampled_window_mask"] = mask.detach()
            updated_inputs["sampled_window_loss_labels"] = loss_labels.detach()
            updated_inputs["sampled_window_token_count"] = window_token_count_tensor.to(raw_reward_full.device)
            self._buffered_inputs[idx] = updated_inputs
            if self.reward_position_tracker.enabled:
                assert position_signals is not None
                position_signals = dict(position_signals)
                position_signals["reward"] = normalized_reward_full
                self.reward_position_tracker.record(position_signals, mask)

        empty_cache()

    def _compute_loss_from_window_cache(self, model, inputs, return_outputs=False):
        for key in (
            "sampled_window_reward",
            "sampled_window_mask",
            "sampled_window_loss_labels",
            "sampled_window_token_count",
        ):
            if key not in inputs:
                raise ValueError(f"Missing {key}; sampled reward window precompute did not run.")

        self._maybe_dump_prefill_debug(
            inputs,
            student_input_ids=inputs["input_ids"],
            student_attention_mask=inputs.get("attention_mask"),
            student_labels=inputs.get("labels"),
            teacher_input_ids=inputs["input_ids"],
            teacher_attention_mask=inputs.get("attention_mask"),
            teacher_labels=inputs.get("labels"),
            student_prompt_length=inputs["prompts"].shape[1],
            teacher_prompt_length=inputs["prompts"].shape[1],
        )
        outputs_student = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        prompt_length = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_length - 1 : -1, :]
        effective_mask = inputs["sampled_window_mask"].to(shifted_student_logits.device).bool()
        loss_labels = inputs["sampled_window_loss_labels"].to(shifted_student_logits.device)
        safe_loss_labels = loss_labels.masked_fill(~effective_mask, 0).long()
        student_log_probs = torch.log_softmax(shifted_student_logits.float(), dim=-1)
        loss_logp_full = student_log_probs.gather(-1, safe_loss_labels.unsqueeze(-1)).squeeze(-1)
        reward_full = inputs["sampled_window_reward"].to(shifted_student_logits.device).detach()

        if not effective_mask.any():
            loss = outputs_student.logits.sum() * 0.0
        elif self.token_loss_normalization == "window_token_mean":
            denominator = inputs["sampled_window_token_count"].to(shifted_student_logits.device).float().clamp_min(1.0)
            loss = -(reward_full[effective_mask] * loss_logp_full[effective_mask]).sum() / denominator
            ga_steps = float(getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps))
            loss = loss * ga_steps
        else:
            loss = -(reward_full[effective_mask] * loss_logp_full[effective_mask]).mean()

        empty_cache()
        return (loss, outputs_student) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_uld_loss:
            raise ValueError("OPDDynamicsTrainer sampled reward loss does not support use_uld_loss=True.")
        if getattr(self.args, "trajectory_source", "student") != "student":
            raise ValueError("OPDDynamicsTrainer expects trajectory_source='student' for on-policy sampled rewards.")
        if self._uses_window_reward_cache():
            return self._compute_loss_from_window_cache(model, inputs, return_outputs=return_outputs)

        self._maybe_dump_prefill_debug(
            inputs,
            student_input_ids=inputs["input_ids"],
            student_attention_mask=inputs.get("attention_mask"),
            student_labels=inputs.get("labels"),
            teacher_input_ids=inputs["input_ids"],
            teacher_attention_mask=inputs.get("attention_mask"),
            teacher_labels=inputs.get("labels"),
            student_prompt_length=inputs["prompts"].shape[1],
            teacher_prompt_length=inputs["prompts"].shape[1],
        )
        outputs_student = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)

        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        prompt_length = inputs["prompts"].shape[1]
        components = self._build_reward_components(
            inputs,
            outputs_student.logits[:, prompt_length - 1 : -1, :],
            outputs_teacher.logits[:, prompt_length - 1 : -1, :],
        )
        effective_mask = components["effective_mask"]
        self._record_terminal_stop_stats(components["terminal_stats"])

        if not effective_mask.any():
            loss = outputs_student.logits.sum() * 0.0
            empty_cache()
            return (loss, outputs_student) if return_outputs else loss

        reward_full = components["reward_full"]
        reward = reward_full[effective_mask].detach()
        loss_logp = components["loss_logp_full"][effective_mask]
        loss = -(reward * loss_logp).mean()

        if self.reward_position_tracker.enabled:
            position_signals = build_reward_position_signals(
                reward_full.detach(),
                components["student_log_probs"],
                components["teacher_log_probs"],
            )
            self.reward_position_tracker.record(position_signals, effective_mask)

        log_ratio = components["log_ratio_full"][effective_mask]
        log_clip = components["log_clip_full"][effective_mask]
        log_tanh = components["log_tanh_full"][effective_mask]
        prob_diff = components["prob_diff_full"][effective_mask]
        sqrt_diff = components["sqrt_diff_full"][effective_mask]
        power_diff = components["power_diff_full"][effective_mask]
        token_entropy = components["token_entropy_full"][effective_mask]
        student_prob = components["student_logp_full"][effective_mask].exp()

        self._summarize_tensor("train_reward", reward)
        self._summarize_tensor("train_log_ratio", log_ratio)
        self._summarize_tensor("train_log_clip", log_clip)
        self._summarize_tensor("train_log_tanh", log_tanh)
        self._summarize_tensor("train_prob_diff", prob_diff)
        self._summarize_tensor("train_sqrt_diff", sqrt_diff)
        self._summarize_tensor("train_power_diff", power_diff)
        self._record_train_scalar("train_policy_entropy", token_entropy.mean())
        self._record_train_scalar("train_student_sampled_prob_mean", student_prob.mean())
        self._record_train_scalar("train_abs_prob_diff_gt_0.01", (prob_diff.abs() > 0.01).float().mean())
        self._record_train_scalar("train_abs_sqrt_diff_gt_0.01", (sqrt_diff.abs() > 0.01).float().mean())
        self._record_train_scalar("train_abs_power_diff_gt_0.01", (power_diff.abs() > 0.01).float().mean())
        self._record_train_scalar("train_abs_log_ratio_gt_0.1", (log_ratio.abs() > 0.1).float().mean())
        self._record_train_scalar("train_abs_log_clip_gt_0.1", (log_clip.abs() > 0.1).float().mean())
        self._record_train_scalar("train_abs_log_tanh_gt_0.1", (log_tanh.abs() > 0.1).float().mean())

        empty_cache()
        return (loss, outputs_student) if return_outputs else loss
