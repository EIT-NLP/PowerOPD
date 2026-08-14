from __future__ import annotations

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

DUAL_EOS_TOKENS = ("<|endoftext|>", "<|im_end|>")


def resolve_dual_eos_stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Resolve the two Qwen stop tokens shared by the base student and chat teacher."""

    token_ids: list[int] = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in DUAL_EOS_TOKENS:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if not isinstance(token_id, int) or token_id < 0:
            continue
        if unk_token_id is not None and token_id == unk_token_id:
            continue
        if token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


def apply_dual_eos_stop_token_ids(trainer) -> list[int]:
    """Attach dual EOS stop ids to the trainer's vLLM generation kwargs."""

    stop_token_ids = resolve_dual_eos_stop_token_ids(trainer.processing_class)
    trainer.dual_eos_stop_token_ids = stop_token_ids
    if stop_token_ids and getattr(trainer, "use_vllm", False) and hasattr(trainer, "vllm_generation"):
        generation_kwargs = dict(getattr(trainer.vllm_generation, "generation_kwargs", {}) or {})
        generation_kwargs["stop_token_ids"] = stop_token_ids
        trainer.vllm_generation.generation_kwargs = generation_kwargs
    return stop_token_ids
