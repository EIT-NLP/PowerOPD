from __future__ import annotations

import warnings
from typing import Any

from datasets import Dataset
import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from trl.experimental.gold.gold_trainer import GOLDTrainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import pad, split_tensor_dict
from trl.experimental.utils import DataCollatorForChatML

from .eval_callback import FactorizedVLLMMathEvalCallback
from .math_eval_utils import (
    MathVerifier,
    compute_rollout_metric_totals,
    empty_rollout_metric_totals,
    finalize_rollout_metric_totals,
    group_flat_rollout_samples,
    supports_enable_thinking,
)


class FactorizedDataCollatorForChatML(DataCollatorForChatML):
    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[Any]]:
        batch = super().__call__(examples)
        if any("answer" in example for example in examples):
            batch["answer"] = [example.get("answer") for example in examples]
        return batch


class FactorizedGOLDTrainer(GOLDTrainer):
    def __init__(self, *args, **kwargs):
        if kwargs.get("data_collator") is None and kwargs.get("processing_class") is not None:
            kwargs["data_collator"] = FactorizedDataCollatorForChatML(
                tokenizer=kwargs["processing_class"],
                max_length=getattr(kwargs.get("args"), "max_length", None),
            )
        super().__init__(*args, **kwargs)
        # GOLDTrainer.compute_loss accepts num_items_in_batch but does not use it,
        # so let Trainer handle gradient-accumulation scaling.
        self.model_accepts_loss_kwargs = False
        self.trajectory_source = self.args.trajectory_source
        self._rollout_verifier = MathVerifier()
        self._train_rollout_metric_totals = empty_rollout_metric_totals()
        has_eval_callback_data = bool(
            getattr(self.args, "eval_test_names", "") and getattr(self.args, "eval_test_paths", "")
        )
        if has_eval_callback_data:
            if self.use_vllm:
                self.add_callback(FactorizedVLLMMathEvalCallback(self))
            else:
                warnings.warn(
                    "eval_test_names/eval_test_paths are set but use_vllm=False. Eval callback is skipped.",
                    stacklevel=2,
                )

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self._signature_columns is None:
            self._signature_columns = []
        if "answer" not in self._signature_columns:
            self._signature_columns.append("answer")

    def _runtime_chat_template_kwargs(self, processing_class, args) -> dict[str, Any]:
        enable_thinking = getattr(args, "enable_thinking", None)
        if enable_thinking is None or not supports_enable_thinking(processing_class):
            return {}
        return {"enable_thinking": enable_thinking}

    def _prepare_dataset_with_original_text(self, dataset, processing_class, args, packing, formatting_func, dataset_name):
        template_kwargs = self._runtime_chat_template_kwargs(processing_class, args)
        if template_kwargs:
            map_kwargs = {}
            if isinstance(dataset, Dataset):
                map_kwargs["num_proc"] = args.dataset_num_proc
                map_kwargs["desc"] = f"Injecting chat template kwargs into {dataset_name} dataset"

            def add_chat_template_kwargs(example):
                example["chat_template_kwargs"] = dict(template_kwargs)
                return example

            dataset = dataset.map(add_chat_template_kwargs, **map_kwargs)

        return super()._prepare_dataset_with_original_text(
            dataset,
            processing_class,
            args,
            packing,
            formatting_func,
            dataset_name,
        )

    def _fill_buffer(self, generation_batch: dict[str, torch.Tensor | Any], buffer_steps: int):
        slices = split_tensor_dict(generation_batch, buffer_steps)
        if self.trajectory_source == "dataset":
            on_policy_flags = [False] * buffer_steps
        else:
            on_policy_flags = [True] * buffer_steps

        self._buffered_inputs = [None] * buffer_steps
        self._buffered_on_policy = on_policy_flags
        self._buffered_text_logs = [None] * buffer_steps

        for i, flag in enumerate(on_policy_flags):
            if not flag:
                slice_inputs = slices[i]
                if self.trajectory_source == "dataset":
                    slice_inputs = self._ensure_original_text_fields(slice_inputs)
                    self._collect_dataset_train_rollout_metrics(slice_inputs)
                if self.use_uld_loss and self.teacher_tokenizer is not None:
                    if self.trajectory_source != "dataset":
                        slice_inputs = self._ensure_original_text_fields(slice_inputs)
                    if "original_prompt_text" not in slice_inputs or "original_completion_text" not in slice_inputs:
                        raise ValueError("Off-policy batch missing original text fields required for ULD alignment.")
                self._buffered_inputs[i] = slice_inputs

        on_policy_indices = [i for i, flag in enumerate(on_policy_flags) if flag]
        if not on_policy_indices:
            return
        if self.trajectory_source == "student":
            self._generate_on_policy_for_slices(slices, on_policy_indices)
        elif self.trajectory_source == "teacher":
            self._generate_teacher_for_slices(slices, on_policy_indices)
        else:
            raise ValueError(f"Unsupported trajectory_source: {self.trajectory_source}")

    def _strip_bos_from_prompt_ids(self, prompt_ids: list[int]) -> list[int]:
        bos_token_id = getattr(self.processing_class, "bos_token_id", None)
        if bos_token_id is not None and prompt_ids and prompt_ids[0] == bos_token_id:
            return prompt_ids[1:]
        return prompt_ids

    def _reset_train_rollout_metric_totals(self):
        self._train_rollout_metric_totals = empty_rollout_metric_totals()

    def _accumulate_train_rollout_metric_totals(self, totals: dict[str, float]):
        for key in self._train_rollout_metric_totals:
            self._train_rollout_metric_totals[key] += float(totals.get(key, 0.0))

    def _normalize_rollout_gold_answer(self, answer) -> str | None:
        if answer is None:
            return None
        normalized = answer.strip() if isinstance(answer, str) else str(answer).strip()
        return normalized or None

    def _decode_prompt_texts_from_slice(self, slice_inputs: dict[str, torch.Tensor | Any]) -> list[str] | None:
        prompts = slice_inputs.get("prompts")
        if prompts is None or not isinstance(prompts, torch.Tensor):
            return None

        prompt_attention_mask = slice_inputs.get("prompt_attention_mask")
        prompt_ids: list[list[int]] = []
        for prompt_idx, prompt in enumerate(prompts):
            if prompt_attention_mask is not None:
                prompt = prompt[prompt_attention_mask[prompt_idx].bool()]
            prompt_ids.append(prompt.tolist())

        return self.processing_class.batch_decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _completion_token_ids_from_labels(self, slice_inputs: dict[str, torch.Tensor | Any]) -> list[list[int]] | None:
        labels = slice_inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            return None
        return [row[row != -100].tolist() for row in labels]

    def _collect_dataset_train_rollout_metrics(self, slice_inputs: dict[str, torch.Tensor | Any]):
        prompt_texts = self._decode_prompt_texts_from_slice(slice_inputs)
        if prompt_texts is None:
            prompt_texts = slice_inputs.get("original_prompt_text")
        completion_token_ids = self._completion_token_ids_from_labels(slice_inputs)
        if prompt_texts is None or completion_token_ids is None:
            return

        raw_answers = slice_inputs.get("answer")
        if raw_answers is None:
            gold_answers = [None] * len(completion_token_ids)
        else:
            gold_answers = [self._normalize_rollout_gold_answer(answer) for answer in raw_answers]

        self._collect_train_rollout_metrics(
            list(prompt_texts),
            completion_token_ids,
            gold_answers,
            include_accuracy=True,
        )

    def _training_clip_max_tokens(self) -> int:
        clip_max_tokens = getattr(self.args, "max_completion_length", None)
        if clip_max_tokens is None:
            clip_max_tokens = getattr(self.generation_config, "max_new_tokens", 0)
        return int(clip_max_tokens)

    def _collect_train_rollout_metrics(
        self,
        prompt_texts: list[str],
        completion_token_ids: list[list[int]],
        gold_answers: list[str | None],
        *,
        include_accuracy: bool,
    ):
        if not prompt_texts or not completion_token_ids:
            return

        grouped_prompts, grouped_golds, grouped_candidates = group_flat_rollout_samples(
            prompt_texts,
            completion_token_ids,
            self.processing_class,
            group_size=max(1, int(self.num_generations)),
            gold_answers=gold_answers,
        )
        if not grouped_candidates:
            return

        model_for_entropy = self.accelerator.unwrap_model(self.model)
        totals = compute_rollout_metric_totals(
            verifier=self._rollout_verifier,
            gold_answers=grouped_golds,
            grouped_candidates=grouped_candidates,
            metric_k=max(1, int(self.num_generations)),
            clip_max_tokens=self._training_clip_max_tokens(),
            model=model_for_entropy,
            tokenizer=self.processing_class,
            prompt_texts=grouped_prompts,
            include_accuracy=include_accuracy,
        )
        self._accumulate_train_rollout_metric_totals(totals)

    def _finalize_train_rollout_logs(self) -> dict[str, float]:
        totals = self._train_rollout_metric_totals
        if not any(value > 0 for value in totals.values()):
            return {}

        keys = list(totals.keys())
        device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
        vec = torch.tensor([totals[key] for key in keys], dtype=torch.float64, device=device)
        if (
            getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)

        reduced_totals = {key: float(value) for key, value in zip(keys, vec.tolist(), strict=True)}
        include_accuracy = self.trajectory_source in {"student", "dataset"}
        metrics = finalize_rollout_metric_totals(reduced_totals, include_accuracy=include_accuracy)
        if self.trajectory_source == "teacher":
            metrics = {"rollout_entropy": metrics["rollout_entropy"]} if "rollout_entropy" in metrics else {}
        self._reset_train_rollout_metric_totals()
        return metrics

    def _generate_on_policy_for_slices(self, slices, on_policy_indices):
        prompt_ids_list = []
        vllm_prompt_ids_list = []
        local_slice_indices = []
        gold_answers: list[str | None] = []
        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            prompt_attention_mask = slice_inputs.get("prompt_attention_mask")
            slice_answers = slice_inputs.get("answer")
            for prompt_idx, prompt in enumerate(slice_inputs["prompts"]):
                if prompt_attention_mask is not None:
                    prompt = prompt[prompt_attention_mask[prompt_idx].bool()]
                prompt_ids = prompt.tolist()
                prompt_ids_list.append(prompt_ids)
                vllm_prompt_ids_list.append(self._strip_bos_from_prompt_ids(prompt_ids))
                local_slice_indices.append(slice_idx)
                answer = None
                if slice_answers is not None and prompt_idx < len(slice_answers):
                    answer = self._normalize_rollout_gold_answer(slice_answers[prompt_idx])
                gold_answers.append(answer)

        prompts_text = self.processing_class.batch_decode(
            prompt_ids_list,
            skip_special_tokens=True,
        )
        prompts_text_with_special = self.processing_class.batch_decode(
            prompt_ids_list,
            skip_special_tokens=False,
        )

        if not self.use_vllm:
            self._generate_non_vllm_for_slices(slices, on_policy_indices)
            return

        if (
            self.state.global_step != self._last_vllm_sync_step
            and self.state.global_step >= self._last_vllm_sync_step + self.vllm_sync_frequency
        ):
            self.vllm_generation.sync_weights()
            self._last_vllm_sync_step = self.state.global_step

        _, completion_ids, _, _ = self.vllm_generation.generate(
            prompts=vllm_prompt_ids_list,
            images=None,
            num_generations=self.num_generations,
        )

        self._collect_train_rollout_metrics(
            prompts_text_with_special,
            [list(ids) for ids in completion_ids],
            gold_answers,
            include_accuracy=True,
        )

        self._process_completions_to_buffer(
            slices,
            on_policy_indices,
            local_slice_indices,
            completion_ids,
            prompt_ids_list,
            prompts_text_with_special,
            prompts_text,
            self.generation_config.max_new_tokens,
        )

    def _generate_teacher_for_slices(self, slices, on_policy_indices):
        flat_prompt_texts: list[str] = []
        flat_completion_token_ids: list[list[int]] = []
        gold_answers: list[str | None] = []
        with unwrap_model_for_generation(
            self.teacher_model,
            self.accelerator,
            generation_kwargs=self.generation_kwargs,
        ) as unwrapped_teacher:
            for slice_idx in on_policy_indices:
                slice_inputs = slices[slice_idx]
                result = self.generate_on_policy_outputs(
                    unwrapped_teacher,
                    slice_inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
                new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts = result

                updated_slice = dict(slice_inputs)
                updated_slice["input_ids"] = new_input_ids
                updated_slice["attention_mask"] = new_attention_mask
                updated_slice["labels"] = new_labels
                updated_slice["original_prompt_text"] = prompt_texts
                updated_slice["original_completion_text"] = completion_texts

                self._buffered_inputs[slice_idx] = updated_slice
                self._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)

                slice_answers = slice_inputs.get("answer")
                for row_idx, prompt_text in enumerate(prompt_texts):
                    flat_prompt_texts.append(prompt_text)
                    label_row = new_labels[row_idx]
                    completion_ids = label_row[label_row != -100].detach().cpu().tolist()
                    flat_completion_token_ids.append([int(token_id) for token_id in completion_ids])
                    answer = None
                    if slice_answers is not None and row_idx < len(slice_answers):
                        answer = self._normalize_rollout_gold_answer(slice_answers[row_idx])
                    gold_answers.append(answer)

        self._collect_train_rollout_metrics(
            flat_prompt_texts,
            flat_completion_token_ids,
            gold_answers,
            include_accuracy=False,
        )

    def _process_completions_to_buffer(
        self,
        slices,
        on_policy_indices,
        local_slice_indices,
        completion_ids,
        prompt_ids_list,
        prompts_text_with_special,
        prompts_text,
        max_completion_length,
    ):
        super()._process_completions_to_buffer(
            slices,
            on_policy_indices,
            local_slice_indices,
            completion_ids,
            prompt_ids_list,
            prompts_text_with_special,
            prompts_text,
            max_completion_length,
        )

        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
        prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        truncation_side = getattr(self.processing_class, "truncation_side", "right")

        grouped_prompt_ids: dict[int, list[list[int]]] = {idx: [] for idx in on_policy_indices}
        for prompt_ids, slice_idx in zip(prompt_ids_list, local_slice_indices, strict=True):
            grouped_prompt_ids[slice_idx].append(prompt_ids)

        for slice_idx in on_policy_indices:
            prompt_tensors = []
            prompt_attention_masks = []
            for prompt_ids in grouped_prompt_ids[slice_idx]:
                if prompt_max_length and len(prompt_ids) > prompt_max_length:
                    if truncation_side == "left":
                        prompt_ids = prompt_ids[-prompt_max_length:]
                    else:
                        prompt_ids = prompt_ids[:prompt_max_length]
                prompt_tensors.append(torch.tensor(prompt_ids, device=device, dtype=torch.long))
                prompt_attention_masks.append(torch.ones(len(prompt_ids), device=device, dtype=torch.long))

            prompt_ids = pad(prompt_tensors, padding_side="left", padding_value=pad_token_id)
            prompt_attention_mask = pad(prompt_attention_masks, padding_side="left", padding_value=0)

            updated_slice = dict(self._buffered_inputs[slice_idx])
            updated_slice["prompts"] = prompt_ids
            updated_slice["prompt_attention_mask"] = prompt_attention_mask
            self._buffered_inputs[slice_idx] = updated_slice

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        merged_logs = dict(logs)
        is_eval_log = any(key.startswith("eval_") for key in merged_logs)
        if self.model.training and not is_eval_log:
            merged_logs.update(self._finalize_train_rollout_logs())
        super().log(merged_logs, start_time)

