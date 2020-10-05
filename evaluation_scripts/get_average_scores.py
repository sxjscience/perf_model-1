import argparse
import glob
import os
import pandas as pd


parser = argparse.ArgumentParser(description='Read model results.')
parser.add_argument('--dir_path', type=str)
parser.add_argument('--out_path', type=str)
args = parser.parse_args()

score_columns = ['spearman', 'ndcg_2', 'ndcg_5', 'ndcg_8']

out = []
for file_path in glob.glob(os.path.join(args.dir_path, '*.csv')):
    model_name = file_path[:-len('.csv')]
    df = pd.read_csv(file_path)
    file_results = []
    for col in score_columns:
        avg_score = df[col].mean()
        file_results.append(avg_score)
    out.append(file_results)
out_df = pd.DataFrame(out, columns=score_columns)
out_df.to_csv(args.out_path)
