import pandas as pd

# Path to the parquet file
parquet_path = "datasets/gsm8k_long_cot/train.parquet"

# Read the parquet file into a pandas DataFrame
df = pd.read_parquet(parquet_path)

# Display the prompt of the first row
print(df.loc[0, "prompt"])
