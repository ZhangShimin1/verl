"""
Preprocess the MATH500 dataset to parquet format (MATH500 with long CoT)
"""

import argparse
import json
import os

import pandas as pd

from verl.utils.hdfs_io import makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="datasets/math500_long_cot")
    parser.add_argument("--data_source", default="datasets/origin_data/math500/test.jsonl")

    args = parser.parse_args()

    data_source = args.data_source

    system_prompt = r"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant should 1) Identify core concepts and required formulas. 2) Break down solutions into logical, numbered steps. 3) Verify results using alternative methods or substitutions. Put your final answer within \boxed{}."  # noqa: E501

    rows = []
    with open(data_source) as f:
        for idx, line in enumerate(f):
            example = json.loads(line)
            question_raw = example.pop("problem")
            answer_raw = example.pop("answer")
            solution_raw = example.pop("solution")
            row = {
                "data_source": "math500_long_cot",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_raw},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            rows.append(row)

    test_dataset = pd.DataFrame(rows)

    local_dir = args.local_dir
    makedirs(local_dir, exist_ok=True)

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
