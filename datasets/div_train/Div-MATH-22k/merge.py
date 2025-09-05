from huggingface_hub import HfApi
import os
import pyarrow.parquet as pq
import pyarrow as pa


def merge_common_columns(path_a: str, path_b: str) -> pa.Table:
    table_a = pq.read_table(path_a)
    table_b = pq.read_table(path_b)

    cols_a = set(table_a.column_names)
    cols_b = set(table_b.column_names)
    common_cols = sorted(list(cols_a & cols_b))
    if not common_cols:
        raise ValueError("No common columns found between input Parquet files.")

    a_sel = table_a.select(common_cols)
    b_sel = table_b.select(common_cols)

    # Align schemas (types/order) before concat
    unified_schema = pa.unify_schemas([a_sel.schema, b_sel.schema])
    a_cast = a_sel.cast(unified_schema)
    b_cast = b_sel.cast(unified_schema)

    merged = pa.concat_tables([a_cast, b_cast], promote=True)
    print(f"Merging columns: {common_cols}")
    return merged


def main():
    src_a = "/home/smzhang/work25/verl/datasets/div_train/MATH-Level345-train/math_train_hard.parquet"
    src_b = "/home/smzhang/work25/verl/datasets/div_train/UniqueDAPO-MATH-17K/math_dapo_unique_17k.parquet"

    out_dir = "/home/smzhang/work25/verl/datasets/div_train/Div-MATH-22k"
    out_path = f"{out_dir}/div-math-22k.parquet"
    os.makedirs(out_dir, exist_ok=True)

    table = merge_common_columns(src_a, src_b)
    pq.write_table(table, out_path)
    print(f"Merged rows: {table.num_rows}")

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
        folder_path=out_dir,
        repo_id="SteveZ25/Div-MATH-22k",
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
