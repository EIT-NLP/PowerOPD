from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def convert_train_dataset(src: Path, dst: Path) -> int:
    with src.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Expected training source to be a single JSON list.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dst.open("w", encoding="utf-8") as f:
        for item in records:
            problem = str(item.get("problem", "")).strip()
            solution = str(item.get("solution", "")).strip()
            answer = str(item.get("answer", "")).strip()
            assistant_text = solution or answer
            if not problem or not assistant_text:
                continue
            row = {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": assistant_text},
                ],
                "answer": answer,
                "has_detailed_solution": bool(solution),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def copy_eval_dataset(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    with dst.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-src", required=True)
    parser.add_argument("--train-dst", required=True)
    parser.add_argument("--eval-src", required=True)
    parser.add_argument("--eval-dst", required=True)
    args = parser.parse_args()

    train_count = convert_train_dataset(Path(args.train_src), Path(args.train_dst))
    eval_count = copy_eval_dataset(Path(args.eval_src), Path(args.eval_dst))
    print(json.dumps({"train_count": train_count, "eval_count": eval_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
