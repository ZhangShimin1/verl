import json
import os
import re
from typing import Any, Dict, Iterable, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Input JSONL path defined per instructions
math_train_path = "origin_data/math/train.jsonl"

# Output path (Parquet)
output_parquet_path = "/home/smzhang/work25/verl/datasets/div_train/math_train_hard.parquet"


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
REWARD_STYLE = "rule-lighteval/MATH_v2"
ALLOWED_LEVELS = {"Level 3", "Level 4", "Level 5"}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_answer(record: Dict[str, Any]) -> str:
    # Prefer explicit 'answer' if present
    if isinstance(record.get("answer"), str) and record["answer"].strip():
        return record["answer"].strip()

    # Fallback: try to parse from 'solution' by grabbing last \\boxed{...}
    solution = record.get("solution", "") or ""
    matches = re.findall(r"\\boxed\{([^}]*)\}", solution)
    if matches:
        return matches[-1].strip()

    # Last resort: use full solution (not ideal, but better than empty)
    return solution.strip()


def to_dcpo(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    level_str = str(record.get("level", "")).strip()
    # Normalize level like "Level 3" -> "3" if possible
    level_num_match = re.search(r"(\d+)", level_str)
    level_num = level_num_match.group(1) if level_num_match else ""

    problem_text = str(record.get("problem", "")).strip()
    ground_truth = extract_answer(record)

    index_str = record.get("id") or record.get("index") or f"math/{level_num}/{idx}.json"

    return {
        "data_source": f"math_level{level_num}" if level_num else "math_level",
        "prompt": [
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"content": problem_text, "role": "user"},
        ],
        "ability": "MATH",
        "reward_model": {
            "ground_truth": ground_truth,
            "style": REWARD_STYLE,
        },
        "extra_info": {"index": index_str},
        "index": index_str,
    }


def filter_and_convert(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        level = str(r.get("level", "")).strip()
        if level in ALLOWED_LEVELS:
            output.append(to_dcpo(r, i))
    return output


def write_parquet(path: str, rows: List[Dict[str, Any]]) -> None:
    # Preserve nested structures using pyarrow so they are not double-escaped
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> None:
    records = read_jsonl(math_train_path)
    dcpo_rows = filter_and_convert(records)
    write_parquet(output_parquet_path, dcpo_rows)
    print(f"Wrote {len(dcpo_rows)} records to {output_parquet_path}")


if __name__ == "__main__":
    main()