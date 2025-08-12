# Original Datasets

All of the original datasets are in `$VERL_ROOT/datasets/origin_data`

# Prompt Styles

## Long CoT

### Example

```python
messages = [
    {
        "role": "system",
        "content": r"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant should 1) Identify core concepts and required formulas. 2) Break down solutions into logical, numbered steps. 3) Verify results using alternative methods or substitutions. Put your final answer within \boxed{}."
    },
    {
        "role": "user",
        "content": question,
    }
]
```

#### gsm8k

**Check parse_args() first!**

```bash
python gsm8k_long_cot.py
```

#### math500

**Check parse_args() first!**

```bash
python math500_long_cot.py
```

## Other Prompts

TODO