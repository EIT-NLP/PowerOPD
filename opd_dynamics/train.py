from __future__ import annotations

import os

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from .config import OPDDynamicsConfig
from .teacher_quantization import get_teacher_quantization_config
from .trainer import OPDDynamicsTrainer


def load_any_dataset(dataset_name: str, dataset_config: str | None, split_name: str):
    if os.path.exists(dataset_name):
        ext = os.path.splitext(dataset_name)[1].lower()
        if ext in {".json", ".jsonl"}:
            return load_dataset("json", data_files=dataset_name, split="train")
        raise ValueError(f"Unsupported local dataset extension: {ext}")
    dataset = load_dataset(dataset_name, name=dataset_config)
    return dataset[split_name]


def main():
    parser = TrlParser((ScriptArguments, OPDDynamicsConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_torch_dtype = training_args.teacher_dtype or model_args.dtype
    teacher_quantization_config = get_teacher_quantization_config(training_args)
    teacher_model_kwargs = dict(
        revision=training_args.teacher_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=teacher_torch_dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if teacher_quantization_config is not None else None,
        quantization_config=teacher_quantization_config,
    )
    if training_args.teacher_model_init_kwargs is not None:
        teacher_model_kwargs.update(training_args.teacher_model_init_kwargs)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_any_dataset(
        script_args.dataset_name,
        script_args.dataset_config,
        script_args.dataset_train_split,
    )
    eval_dataset = None
    if training_args.eval_strategy != "no" and getattr(script_args, "dataset_test_split", None):
        try:
            eval_dataset = load_any_dataset(
                script_args.dataset_name,
                script_args.dataset_config,
                script_args.dataset_test_split,
            )
        except Exception:
            eval_dataset = None

    trainer = OPDDynamicsTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
