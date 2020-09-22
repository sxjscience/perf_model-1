import pandas as pd
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Read model results.')
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
columns = ['name', 'rmse', 'mae', 'ndcg_all', 'ndcg_k3_all', 'mrr_all',
           'ndcg_valid', 'ndcg_k3_valid', 'mrr_valid']
results = []
for dir_name in os.listdir(args.dir_path):
    for exp_name in os.listdir(os.path.join(args.dir_path, dir_name)):
        with open(os.path.join(args.dir_path, dir_name, exp_name, 'test_scores.json'), 'r') as in_f:
            dat = json.load(in_f)
            results.append(('f{dir_name}/f{exp_name}',
                            dat['rmse'], dat['mae'],
                            dat['ndcg_all'], dat['ndcg_k3_all'], dat['mrr_all'],
                            dat['ndcg_valid'], dat['ndcg_k3_valid'], dat['mrr_valid']))
df = pd.DataFrame(results, columns=columns)
df.to_csv('f{args.model_name}.csv')
