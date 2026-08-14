from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .math_eval_utils import supports_enable_thinking


def parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}.")


def load_any_dataset(dataset_name: str, dataset_config: str | None, split_name: str):
    if os.path.exists(dataset_name):
        ext = os.path.splitext(dataset_name)[1].lower()
        if ext in {".json", ".jsonl"}:
            return load_dataset("json", data_files=dataset_name, split="train")
        raise ValueError(f"Unsupported local dataset extension: {ext}")
    dataset = load_dataset(dataset_name, name=dataset_config)
    return dataset[split_name]


def build_messages_prompt(tokenizer, messages: list[dict], enable_thinking: bool | None) -> tuple[str, list[dict]]:
    prompt_messages = [dict(message) for message in messages if message.get("role") != "assistant"]
    if not prompt_messages:
        raise ValueError("Expected at least one non-assistant message to build the teacher prompt.")

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if supports_enable_thinking(tokenizer) and enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prompt = tokenizer.apply_chat_template(prompt_messages, **kwargs)
    bos_token = getattr(tokenizer, "bos_token", None)
    if isinstance(bos_token, str) and bos_token and prompt.startswith(bos_token):
        prompt = prompt[len(bos_token) :]
    return prompt, prompt_messages


def count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fin:
        return sum(1 for _ in fin)


def emit_progress(*, completed: int, total: int, output_path: Path, resumed: bool):
    print(
        json.dumps(
            {
                "status": "progress",
                "completed": completed,
                "total": total,
                "output_path": str(output_path),
                "resumed": resumed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--teacher_model_name_or_path", required=True)
    parser.add_argument("--trust_remote_code", default="True")
    parser.add_argument("--enable_thinking", default="False")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    trust_remote_code = bool(parse_bool(args.trust_remote_code))
    enable_thinking = parse_bool(args.enable_thinking)

    input_dataset = load_any_dataset(args.input_path, args.dataset_config, args.dataset_split)
    total_records = len(input_dataset)
    if args.limit is not None:
        input_dataset = input_dataset.select(range(min(args.limit, total_records)))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    resume_count = 0
    resumed = False

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if tmp_output_path.exists():
            tmp_output_path.unlink()
    else:
        if tmp_output_path.exists():
            resume_count = count_jsonl_lines(tmp_output_path)
            resumed = resume_count > 0
        elif output_path.exists():
            resume_count = count_jsonl_lines(output_path)
            if resume_count >= len(input_dataset):
                print(
                    json.dumps(
                        {
                            "input_path": args.input_path,
                            "output_path": str(output_path),
                            "generated_count": resume_count,
                            "teacher_model_name_or_path": args.teacher_model_name_or_path,
                            "resumed": False,
                            "already_complete": True,
                        },
                        ensure_ascii=False,
                    )
                )
                return
            output_path.replace(tmp_output_path)
            resumed = resume_count > 0

    if resume_count > len(input_dataset):
        raise ValueError(
            f"Existing output has {resume_count} rows, but current dataset only has {len(input_dataset)} rows."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.teacher_model_name_or_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )

    generated_count = resume_count
    if generated_count:
        emit_progress(completed=generated_count, total=len(input_dataset), output_path=output_path, resumed=resumed)

    open_mode = "a" if resumed else "w"
    with tmp_output_path.open(open_mode, encoding="utf-8") as fout:
        for start in range(resume_count, len(input_dataset), args.batch_size):
            stop = min(start + args.batch_size, len(input_dataset))
            batch = [input_dataset[idx] for idx in range(start, stop)]
            prompts: list[str] = []
            prompt_message_batches: list[list[dict]] = []
            for record in batch:
                prompt, prompt_messages = build_messages_prompt(tokenizer, record["messages"], enable_thinking)
                prompts.append(prompt)
                prompt_message_batches.append(prompt_messages)

            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=(start == resume_count))
            for record, prompt_messages, output in zip(batch, prompt_message_batches, outputs, strict=True):
                completion_text = output.outputs[0].text if output.outputs else ""
                row = dict(record)
                row.pop("chat_template_kwargs", None)
                row["messages"] = prompt_messages + [{"role": "assistant", "content": completion_text}]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                generated_count += 1

            fout.flush()
            emit_progress(completed=generated_count, total=len(input_dataset), output_path=output_path, resumed=resumed)

    tmp_output_path.replace(output_path)
    print(
        json.dumps(
            {
                "input_path": args.input_path,
                "output_path": str(output_path),
                "generated_count": generated_count,
                "teacher_model_name_or_path": args.teacher_model_name_or_path,
                "resumed": resumed,
                "already_complete": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
