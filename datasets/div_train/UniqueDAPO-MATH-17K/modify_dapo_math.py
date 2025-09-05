import json
import os
import re
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


INPUT_PARQUET = "/home/smzhang/work25/verl/datasets/dapo/uniquedapo-math-17k.parquet"
OUTPUT_PARQUET = "/home/smzhang/work25/verl/datasets/div_train/UniqueDAPO-MATH-17K/math_dapo_unique_17k.parquet"

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


INSTRUCTION_PATTERNS = [
    r"^\s*Solve the following math problem step by step[\s\S]*?\n\n",
    r"^\s*Solve the following problem step by step[\s\S]*?\n\n",
    r"The last line of your response should be[\s\S]*?$",
    r"Remember to put your answer[\s\S]*?$",
    r"^\s*Answer:\s*$",
]


def clean_user_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    cleaned = text
    # Remove common instruction prefaces and suffixes
    for pattern in INSTRUCTION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # If still contains a double-newline preface, drop everything up to the last double newline
    if "\n\n" in cleaned:
        parts = [p for p in cleaned.split("\n\n") if p.strip()]
        if parts:
            cleaned = "\n\n".join(parts[1:]) if len(parts) > 1 else parts[0]

    # Trim trailing reminders or boilerplate lines
    lines = [ln for ln in cleaned.splitlines() if ln.strip() and not ln.strip().lower().startswith("remember to ")]
    cleaned = "\n".join(lines).strip()

    # Fallback to original if cleaning erased content
    return cleaned if cleaned else text.strip()


def normalize_prompt(prompt_value: Any) -> List[Dict[str, Any]]:
    # Handle either JSON string or already-parsed list
    prompt_list: List[Dict[str, Any]]
    if isinstance(prompt_value, str):
        try:
            prompt_list = json.loads(prompt_value)
        except Exception:
            prompt_list = []
    elif isinstance(prompt_value, list):
        prompt_list = prompt_value
    else:
        prompt_list = []

    # Prefer first user message content
    user_content = ""
    for msg in prompt_list:
        if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
            user_content = msg["content"]
            break

    cleaned_problem = clean_user_content(user_content)

    return [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": cleaned_problem, "role": "user"},
    ]


def process_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)

        # Build new prompt
        new_prompt = normalize_prompt(row.get("prompt"))
        new_row["prompt"] = new_prompt

        # Add raw_problem for convenience
        new_row["raw_problem"] = new_prompt[1]["content"]

        # Optional: drop pandas auto-index column if present
        if "__index_level_0__" in new_row:
            del new_row["__index_level_0__"]

        output.append(new_row)
    return output


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

    table = pq.read_table(INPUT_PARQUET)
    rows: List[Dict[str, Any]] = table.to_pylist()
    processed = process_rows(rows)

    out_table = pa.Table.from_pylist(processed)
    pq.write_table(out_table, OUTPUT_PARQUET)
    print(f"Wrote {len(processed)} rows to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()


