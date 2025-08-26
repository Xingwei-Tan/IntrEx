import pandas as pd
import numpy as np
from datasets import load_dataset
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="XingweiT/IntrEx-sequence")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="train")
    df = dataset.to_pandas()

    pick_cols = []
    for col in df.columns:
        if col.endswith("_exp_int"):
            continue
        elif col.endswith("_int"):
            pick_cols.append(col)
    score_df = df[pick_cols]
    row_avg = score_df.mean(axis=1).tolist()

    

    print(f"Mean scores of INT: {np.mean(row_avg)}")
    print(f"Standard deviation of the scores of INT: {np.std(row_avg)}")

    pick_cols = []
    for col in df.columns:
        if col.endswith("exp_int"):
            pick_cols.append(col)
    score_df = df[pick_cols]
    row_avg = score_df.mean(axis=1).tolist()

    

    print(f"Mean scores of EXP INT: {np.mean(row_avg)}")
    print(f"Standard deviation of the scores of EXP INT: {np.std(row_avg)}")

