import pandas as pd
import numpy as np
from irrCAC.raw import CAC
from datasets import load_dataset
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="XingweiT/IntrEx-sequence")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="train")
    df = dataset.to_pandas()


    ac2_scores = []
    conversation_ids = set()
    print(f"Number of rows in the dataset: {len(df)}")
    projects = list(df['project_id'].unique())
    for p_id in projects:
        project_df = df[df['project_id'] == p_id]
        pick_cols = []
        score_dict = {}
        for col in ["p0", "p1", "p2"]:
            combined_score = project_df[col+"_int"].tolist() + project_df[col+"_exp_int"].tolist()
            score_dict[col] = combined_score
        score_df = pd.DataFrame(score_dict)
        score_df = score_df.dropna()
        ac2 = CAC(score_df, categories=[0,1,2,3,4], weights="linear").gwet()['est']['coefficient_value']
        ac2_scores.append(ac2)
        for i in range(len(df)):
            conversation_ids.add(df['conversation_id'][i])
    print(f"Number of projects: {len(ac2_scores)}")
    print(f"Number of conversations: {len(conversation_ids)}")
    print(f"Mean AC2 scores of INT&EXP INT combined: {np.mean(ac2_scores)}")
    print(f"Standard deviation of the AC2 scores of INT&EXP INT combined: {np.std(ac2_scores)}")


    ac2_scores = []
    for p_id in projects:
        project_df = df[df['project_id'] == p_id]
        pick_cols = []
        for col in df.columns:
            if col.endswith("_exp_int"):
                continue
            elif col.endswith("_int"):
                pick_cols.append(col)
        score_df = project_df[pick_cols]
        score_df = score_df.dropna()
        ac2 = CAC(score_df, categories=[0,1,2,3,4], weights="linear").gwet()['est']['coefficient_value']
        ac2_scores.append(ac2)

    print(f"Mean AC2 scores of INT: {np.mean(ac2_scores)}")
    print(f"Standard deviation of the AC2 scores of INT: {np.std(ac2_scores)}")

    ac2_scores = []
    for p_id in projects:
        project_df = df[df['project_id'] == p_id]
        pick_cols = []
        for col in df.columns:
            if col.endswith("_exp_int"):
                pick_cols.append(col)
        score_df = project_df[pick_cols]
        score_df = score_df.dropna()
        ac2 = CAC(score_df, categories=[0,1,2,3,4], weights="linear").gwet()['est']['coefficient_value']
        ac2_scores.append(ac2)

    print(f"Mean AC2 scores of EXP INT: {np.mean(ac2_scores)}")
    print(f"Standard deviation of the AC2 scores of EXP INT: {np.std(ac2_scores)}")
