from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig


def resolve_torch_dtype(dtype: str | torch.dtype | None, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if dtype is None:
        return default
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return default
    return getattr(torch, str(dtype))


def get_teacher_quantization_config(args) -> BitsAndBytesConfig | None:
    load_in_4bit = getattr(args, "teacher_load_in_4bit", False)
    load_in_8bit = getattr(args, "teacher_load_in_8bit", False)
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of teacher_load_in_4bit and teacher_load_in_8bit can be enabled.")
    if load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(getattr(args, "teacher_bnb_8bit_threshold", 6.0)),
        )
    if not load_in_4bit:
        return None
    compute_dtype = resolve_torch_dtype(getattr(args, "teacher_bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=getattr(args, "teacher_bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=getattr(args, "teacher_use_bnb_nested_quant", False),
    )
