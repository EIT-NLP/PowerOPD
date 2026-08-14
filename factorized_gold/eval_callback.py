from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

from transformers import TrainerCallback

from .stop_tokens import resolve_dual_eos_stop_token_ids

from .math_eval_utils import (
    MathVerifier,
    build_prompt,
    compute_rollout_metric_totals,
    ensure_parent,
    finalize_rollout_metric_totals,
    prepare_eval_dataset,
    score_candidates,
)


class FactorizedVLLMMathEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.args = trainer.args
        self.eval_steps = int(getattr(self.args, "eval_steps", 0) or 0)
        self._last_eval_step = -1
        self._initial_eval_done = False

        names = [x.strip() for x in getattr(self.args, "eval_test_names", "").split(",") if x.strip()]
        paths = [x.strip() for x in getattr(self.args, "eval_test_paths", "").split(",") if x.strip()]
        samples = [x.strip() for x in getattr(self.args, "eval_test_samples", "all").split(",") if x.strip()]
        if len(samples) == 1 and names:
            samples = samples * len(names)
        if len(names) != len(paths) or len(names) != len(samples):
            raise ValueError("eval_test_names, eval_test_paths, and eval_test_samples must have matching counts.")

        self.names = names
        self.paths = paths
        self.samples = samples
        self.verifier = MathVerifier()
        self.enable_thinking = getattr(self.args, "enable_thinking", False)
        self.eval_output_dir = ensure_parent(Path(self.args.output_dir) / "eval_outputs" / "placeholder.txt").parent
        self.all_test_data = {
            name: prepare_eval_dataset(name, path, limit)
            for name, path, limit in zip(self.names, self.paths, self.samples, strict=True)
        }

    def _needs_all_ranks_for_eval(self) -> bool:
        vg = self.trainer.vllm_generation
        return bool(vg.mode == "colocate" and int(getattr(vg, "tensor_parallel_size", 1)) > 1)

    def _sync_weights_if_needed(self):
        vg = self.trainer.vllm_generation
        if getattr(self.trainer, "use_vllm", False):
            vg.sync_weights()
            if hasattr(self.trainer, "_last_vllm_sync_step"):
                self.trainer._last_vllm_sync_step = self.trainer.state.global_step

    def _generate_grouped_candidates(self, prompts: list[str], eval_n: int, eval_max_tokens: int):
        vg = self.trainer.vllm_generation
        is_main = self.trainer.accelerator.is_main_process
        eval_top_p = getattr(self.args, "eval_top_p", getattr(self.args, "top_p", 1.0))
        eval_top_k = getattr(self.args, "eval_top_k", getattr(self.args, "top_k", 0))
        eval_min_p = getattr(self.args, "eval_min_p", getattr(self.args, "min_p", 0.0))
        eval_temp = getattr(self.args, "eval_temperature", 0.0)
        repetition_penalty = getattr(
            self.args,
            "eval_repetition_penalty",
            getattr(self.args, "repetition_penalty", 1.0),
        )
        stop_token_ids = resolve_dual_eos_stop_token_ids(self.trainer.processing_class)
        generation_kwargs = dict(getattr(vg, "generation_kwargs", {}) or {})
        if stop_token_ids:
            generation_kwargs["stop_token_ids"] = stop_token_ids

        if vg.mode == "server":
            if not is_main:
                return []
            response = vg.vllm_client.generate(
                prompts=prompts,
                n=eval_n,
                repetition_penalty=repetition_penalty,
                temperature=eval_temp,
                top_p=eval_top_p,
                top_k=eval_top_k,
                min_p=eval_min_p,
                max_tokens=eval_max_tokens,
                structured_outputs_regex=getattr(vg, "structured_outputs_regex", None),
                generation_kwargs=generation_kwargs,
            )
            completion_ids = response["completion_ids"]
            grouped = []
            idx = 0
            for _ in prompts:
                candidates = []
                for _ in range(eval_n):
                    if idx < len(completion_ids):
                        token_ids = list(completion_ids[idx])
                        text = self.trainer.processing_class.decode(
                            token_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        candidates.append({"text": text, "token_ids": token_ids, "token_len": len(token_ids)})
                    idx += 1
                grouped.append(candidates)
            return grouped

        from vllm import SamplingParams

        if getattr(vg, "enable_sleep_mode", False):
            vg.llm.wake_up(tags=["weights", "kv_cache"])
        sampling_kwargs = {
            "n": eval_n,
            "temperature": eval_temp,
            "top_p": eval_top_p,
            "top_k": eval_top_k,
            "min_p": eval_min_p,
            "repetition_penalty": repetition_penalty,
            "max_tokens": eval_max_tokens,
        }
        if stop_token_ids:
            sampling_kwargs["stop_token_ids"] = stop_token_ids
        sampling_params = SamplingParams(**sampling_kwargs)
        try:
            all_outputs = vg.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=is_main)
            if not is_main:
                return []
            return [
                [
                    {
                        "text": candidate.text,
                        "token_ids": list(candidate.token_ids),
                        "token_len": len(candidate.token_ids),
                    }
                    for candidate in output.outputs
                ]
                for output in all_outputs
            ]
        finally:
            if getattr(vg, "enable_sleep_mode", False):
                vg.llm.sleep(level=2)

    def _do_eval(self, state) -> None:
        if not self.all_test_data or not getattr(self.trainer, "use_vllm", False):
            return
        is_main = self.trainer.accelerator.is_main_process
        needs_all_ranks = self._needs_all_ranks_for_eval()
        if not is_main and not needs_all_ranks:
            return

        self._sync_weights_if_needed()
        details_rows = []
        eval_n = max(1, int(getattr(self.args, "eval_n", 1)))
        eval_k = max(1, int(getattr(self.args, "eval_k", 1)))
        if eval_k > eval_n:
            raise ValueError(f"eval_k ({eval_k}) must be <= eval_n ({eval_n}).")
        eval_max_tokens = getattr(self.args, "eval_max_new_tokens", None)
        if eval_max_tokens is None:
            eval_max_tokens = getattr(self.args, "max_completion_length", 512)

        for name, data in self.all_test_data.items():
            if not data:
                continue
            prompts = [build_prompt(self.trainer.processing_class, item["question"], self.enable_thinking) for item in data]
            grouped_candidates = self._generate_grouped_candidates(prompts, eval_n, eval_max_tokens)
            if not is_main:
                continue

            start_time = time.time()
            for sample_idx, (item, candidates) in enumerate(zip(data, grouped_candidates, strict=True)):
                scored = score_candidates(self.verifier, item["gold"], candidates, eval_k)
                details_rows.append(
                    {
                        "step": int(state.global_step),
                        "dataset_name": name,
                        "sample_idx": int(sample_idx),
                        "question": item["question"],
                        "gold_answer": item["gold"],
                        "prediction_text": scored["selected_text"],
                        "is_correct": bool(scored["is_correct"]),
                        "prediction_token_len": int(scored["selected_len"]),
                        "avg_at_k": float(scored["avg_at_k"]),
                        "pass_at_k": float(scored["pass_at_k"]),
                        "k_used": int(scored["k_used"]),
                        "model_path": str(getattr(self.trainer, "model_name_or_path", "")),
                    }
                )

            model_for_entropy = self.trainer.accelerator.unwrap_model(self.trainer.model)
            totals = compute_rollout_metric_totals(
                verifier=self.verifier,
                gold_answers=[item["gold"] for item in data],
                grouped_candidates=grouped_candidates,
                metric_k=eval_k,
                clip_max_tokens=eval_max_tokens,
                model=model_for_entropy,
                tokenizer=self.trainer.processing_class,
                prompt_texts=prompts,
                include_accuracy=True,
            )
            metrics = finalize_rollout_metric_totals(totals, include_accuracy=True)
            eval_logs = {
                f"eval_{name}_rollout_acc": metrics.get("rollout_acc", 0.0),
                f"eval_{name}_rollout_pass_at_k": metrics.get("rollout_pass_at_k", 0.0),
                f"eval_{name}_rollout_avg_len": metrics.get("rollout_avg_len", 0.0),
                f"eval_{name}_rollout_clip_ratio": metrics.get("rollout_clip_ratio", 0.0),
                f"eval_{name}_rollout_entropy": metrics.get("rollout_entropy", 0.0),
                f"eval_{name}_count": len(data),
                "step": state.global_step,
            }
            self.trainer.log(eval_logs)
            print(
                f"[Factorized GOLD Eval] {name} step={state.global_step} "
                f"acc={eval_logs[f'eval_{name}_rollout_acc']:.4f} "
                f"pass@k={eval_logs[f'eval_{name}_rollout_pass_at_k']:.4f} "
                f"entropy={eval_logs[f'eval_{name}_rollout_entropy']:.4f} "
                f"avg_len={eval_logs[f'eval_{name}_rollout_avg_len']:.1f} "
                f"clip_ratio={eval_logs[f'eval_{name}_rollout_clip_ratio']:.4f} "
                f"n={len(data)} time={time.time() - start_time:.1f}s"
            )

        if is_main:
            details_path = self.eval_output_dir / f"eval_step_{int(state.global_step)}.jsonl"
            try:
                with details_path.open("w", encoding="utf-8") as f:
                    for row in details_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                warnings.warn(f"Failed to write eval details to {details_path}: {exc}", stacklevel=2)

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.all_test_data or self._initial_eval_done:
            return control
        self._do_eval(state)
        self._initial_eval_done = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.all_test_data or self.eval_steps <= 0:
            return control
        if state.global_step <= 0 or state.global_step % self.eval_steps != 0:
            return control
        if state.global_step == self._last_eval_step:
            return control
        self._do_eval(state)
        self._last_eval_step = state.global_step
        return control
