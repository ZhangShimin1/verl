import argparse
from pathlib import Path
from typing import List

import pandas as pd


TARGET_COLUMNS: List[str] = [
    "data_source",
    "prompt",
    "ability",
    "reward_model",
]


def filter_parquet_file(input_path: Path, output_path: Path) -> None:
    """Read a parquet file, ensure target columns exist, keep only them, and write output.

    - Missing columns are added with NA values
    - Column order follows TARGET_COLUMNS
    """
    df = pd.read_parquet(input_path)

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[TARGET_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter parquet files to only the specified columns."
    )
    parser.add_argument(
        "--dir",
        dest="directory",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory containing parquet files (default: this script's directory)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite original files instead of writing filtered_*.parquet",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    parquet_paths = sorted(directory.glob("*.parquet"))

    if not parquet_paths:
        print(f"No parquet files found in {directory}")
        return

    for parquet_path in parquet_paths:
        if args.overwrite:
            output_path = parquet_path
        else:
            output_path = parquet_path.with_name(f"filtered_{parquet_path.name}")

        print(f"Processing {parquet_path.name} -> {output_path.name}")
        filter_parquet_file(parquet_path, output_path)

    print("Done.")


if __name__ == "__main__":
    main()


