from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

STANDARD_TERMINAL_STOP_TARGETS = {"raw", "mask", "im_end"}
VALID_TERMINAL_STOP_TARGETS = STANDARD_TERMINAL_STOP_TARGETS | {"cross_eos"}


@dataclass
class TerminalStopResult:
    labels: torch.Tensor
    mask: torch.Tensor
    stats: dict[str, int]


@dataclass
class CrossEOSTerminalStopResult:
    student_labels: torch.Tensor
    teacher_labels: torch.Tensor
    loss_labels: torch.Tensor
    mask: torch.Tensor
    stats: dict[str, int]


def _token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int | None:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _empty_stats() -> dict[str, int]:
    return {
        "terminal_stop_endoftext_count": 0,
        "terminal_stop_im_end_count": 0,
        "terminal_stop_missing_count": 0,
        "terminal_stop_canonicalized_count": 0,
        "terminal_stop_masked_count": 0,
        "terminal_stop_cross_eos_count": 0,
    }


def resolve_terminal_stop_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int | None, int | None]:
    return _token_id(tokenizer, "<|endoftext|>"), _token_id(tokenizer, "<|im_end|>")


def apply_terminal_stop_target(
    shifted_labels: torch.Tensor,
    mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    target: str,
) -> TerminalStopResult:
    """Apply raw/mask/im_end terminal-stop handling to completion labels only."""

    if target not in STANDARD_TERMINAL_STOP_TARGETS:
        raise ValueError(
            f"apply_terminal_stop_target supports only {sorted(STANDARD_TERMINAL_STOP_TARGETS)}, got {target!r}. "
            "Use apply_cross_eos_terminal_stop_target for `cross_eos`."
        )

    effective_labels = shifted_labels.clone()
    effective_mask = mask.clone()
    endoftext_id, im_end_id = resolve_terminal_stop_ids(tokenizer)
    stats = _empty_stats()

    for row_idx in range(shifted_labels.shape[0]):
        valid_positions = mask[row_idx].nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            stats["terminal_stop_missing_count"] += 1
            continue

        last_pos = valid_positions[-1]
        last_id = int(shifted_labels[row_idx, last_pos].detach().item())
        is_endoftext = endoftext_id is not None and last_id == endoftext_id
        is_im_end = im_end_id is not None and last_id == im_end_id

        if is_endoftext:
            stats["terminal_stop_endoftext_count"] += 1
        elif is_im_end:
            stats["terminal_stop_im_end_count"] += 1
        else:
            stats["terminal_stop_missing_count"] += 1
            continue

        if target == "mask":
            effective_mask[row_idx, last_pos] = False
            stats["terminal_stop_masked_count"] += 1
        elif target == "im_end" and im_end_id is not None:
            if last_id != im_end_id:
                effective_labels[row_idx, last_pos] = im_end_id
                stats["terminal_stop_canonicalized_count"] += 1

    return TerminalStopResult(labels=effective_labels, mask=effective_mask, stats=stats)


def apply_cross_eos_terminal_stop_target(
    shifted_labels: torch.Tensor,
    mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> CrossEOSTerminalStopResult:
    """Align terminal stop semantics without changing the student's action.

    If the sampled final completion token is <|endoftext|>, student/loss labels
    stay on <|endoftext|>, while teacher labels use <|im_end|>. Final <|im_end|>
    already uses <|im_end|> on both sides. Non-stop final tokens are unchanged.
    """

    student_labels = shifted_labels.clone()
    teacher_labels = shifted_labels.clone()
    loss_labels = shifted_labels.clone()
    effective_mask = mask.clone()
    endoftext_id, im_end_id = resolve_terminal_stop_ids(tokenizer)
    stats = _empty_stats()

    for row_idx in range(shifted_labels.shape[0]):
        valid_positions = mask[row_idx].nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            stats["terminal_stop_missing_count"] += 1
            continue

        last_pos = valid_positions[-1]
        last_id = int(shifted_labels[row_idx, last_pos].detach().item())
        is_endoftext = endoftext_id is not None and last_id == endoftext_id
        is_im_end = im_end_id is not None and last_id == im_end_id

        if is_endoftext:
            stats["terminal_stop_endoftext_count"] += 1
            if im_end_id is not None:
                teacher_labels[row_idx, last_pos] = im_end_id
                stats["terminal_stop_cross_eos_count"] += 1
        elif is_im_end:
            stats["terminal_stop_im_end_count"] += 1
        else:
            stats["terminal_stop_missing_count"] += 1

    return CrossEOSTerminalStopResult(
        student_labels=student_labels,
        teacher_labels=teacher_labels,
        loss_labels=loss_labels,
        mask=effective_mask,
        stats=stats,
    )
