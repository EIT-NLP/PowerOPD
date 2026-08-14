from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .math_eval_utils import (
    MathVerifier,
    build_prompt,
    compute_mean_response_entropy,
    ensure_parent,
    prepare_eval_dataset,
    score_candidates,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_name", default="math")
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--eval_samples", default="all")
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--eval_n", type=int, default=1, help="How many responses to sample per prompt during evaluation.")
    parser.add_argument(
        "--eval_k",
        type=int,
        default=1,
        help="How many of the sampled responses are used for metrics. The evaluator reports acc=avg@k and pass=pass@k, and requires eval_n >= eval_k.",
    )
    parser.add_argument("--eval_max_new_tokens", type=int, default=5000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    if args.eval_k < 1 or args.eval_n < 1 or args.eval_k > args.eval_n:
        raise ValueError(f"eval_k ({args.eval_k}) must satisfy 1 <= eval_k <= eval_n ({args.eval_n}).")

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    data = prepare_eval_dataset(args.eval_name, args.eval_path, args.eval_samples)
    prompts = [build_prompt(tokenizer, item["question"], args.enable_thinking) for item in data]
    verifier = MathVerifier()
    llm = LLM(
        model=args.model_name_or_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        n=max(1, args.eval_n),
        temperature=args.eval_temperature,
        max_tokens=args.eval_max_new_tokens,
    )
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    grouped_candidates = [
        [
            {"text": candidate.text, "token_ids": list(candidate.token_ids), "token_len": len(candidate.token_ids)}
            for candidate in output.outputs
        ]
        for output in outputs
    ]

    acc_sum = 0.0
    pass_sum = 0.0
    total_len = 0
    total_candidate_count = 0
    rows = []
    for idx, (item, candidates) in enumerate(zip(data, grouped_candidates, strict=True)):
        scored = score_candidates(verifier, item["gold"], candidates, args.eval_k)
        acc_sum += scored["avg_at_k"]
        pass_sum += scored["pass_at_k"]
        total_len += sum(candidate["token_len"] for candidate in candidates[: scored["k_used"]])
        total_candidate_count += scored["k_used"]
        rows.append(
            {
                "sample_idx": idx,
                "question": item["question"],
                "gold_answer": item["gold"],
                "prediction_text": scored["selected_text"],
                "is_correct": scored["is_correct"],
                "prediction_token_len": scored["selected_len"],
                "avg_at_k": scored["avg_at_k"],
                "pass_at_k": scored["pass_at_k"],
                "k_used": scored["k_used"],
            }
        )

    del llm
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype="auto",
    )
    model = model.to("cuda")
    entropy = compute_mean_response_entropy(model, tokenizer, prompts, grouped_candidates, args.eval_k)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc = acc_sum / max(1, len(data))
    pass_at_k = pass_sum / max(1, len(data))
    avg_len = total_len / max(1, total_candidate_count)
    output_path = ensure_parent(args.output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "eval_name": args.eval_name,
        "count": len(data),
        "acc": acc,
        f"avg_at_{args.eval_k}": acc,
        "pass": pass_at_k,
        f"pass_at_{args.eval_k}": pass_at_k,
        "entropy": entropy,
        "avg_len": avg_len,
        "elapsed_sec": time.time() - start_time,
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
