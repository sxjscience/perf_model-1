# pylint: disable=missing-docstring, invalid-name, ungrouped-imports
# pylint: disable=unsupported-assignment-operation, no-member
import argparse
import logging
import os
import multiprocessing
import json
import torch
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import random
import catboost
import pandas as pd
import tqdm
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from util import analyze_valid_threshold, logging_config, read_pd


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)


def split_df(df, seed, ratio):
    """Split the input data frame into Train + Valid + Test

    Parameters
    ----------
    df
        The input dataset in pandas DataFrame
    seed
        The seed to split the train + remaining
    ratio
        The ratio of the remaining dataset

    Returns
    -------
    train_df
        The training dataframe
    val_df
        The validation dataframe
    """
    rng = np.random.RandomState(seed)
    num = int(ratio * len(df))
    perm = rng.permutation(len(df))
    train_num = len(df) - num
    train_df = df.iloc[perm[:train_num]]
    val_df = df.iloc[perm[train_num:]]
    return train_df, val_df


def split_train_test_df(df, seed, ratio, top_sample_ratio=0.2, group_size=10, K=4):
    """Get the dataframe for testing

    Parameters
    ----------
    df
        The input DataFrame
    seed
        The seed
    ratio
        The ratio to split the testing set
    top_sample_ratio
        The ratio of putting top samples to the test set.
        This tests the ability of the model to predict the latency / ranking of
        the out-of-domain samples.
    group_size
        Num of samples in each group
    K
        For each sample in the test set.
        Sample K groups that contain this test sample.

    Returns
    -------
    train_df
        The training dataframe.
    test_df
        The testing dataframe. This contains samples in the test set,
        and can be used for regression analysis
    test_rank_group_features
        (#samples, #group_size, #features)
    test_rank_group_labels
        (#samples, #group_size)
    """
    rng = np.random.RandomState(seed)
    num_samples = len(df)
    test_num = int(ratio * num_samples)
    train_num = len(df) - test_num
    assert top_sample_ratio > 0
    top_test_num = int(test_num * top_sample_ratio)
    other_test_num = test_num - top_test_num
    if top_test_num > 0:
        # The test samples contain the top throughput samples + random samples
        thrpt = df['thrpt']
        idx = np.argsort(thrpt)
        top_thrpt_indices = idx[-top_test_num:]
        other_thrpt_indices = idx[:(-top_test_num)]
        rng.shuffle(other_thrpt_indices)
        other_test_thrpt_indices = other_thrpt_indices[-other_test_num:]
        test_indices = np.concatenate([top_thrpt_indices, other_test_thrpt_indices], axis=0)
        train_indices = other_thrpt_indices[:(-other_test_num)]
    else:
        top_thrpt_indices = []
        perm = rng.permutation(len(df))
        train_indices = perm[:train_num]
        test_indices = perm[train_num:]
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    # Get ranking dataframe, for each sample in the test set, randomly sample
    # group_size - 1 elements
    test_rank_arr = []
    for i, idx in enumerate(test_indices):
        for _ in range(K):
            group_indices = (rng.choice(num_samples - 1, group_size - 1, True) + idx + 1) % num_samples
            group_indices = np.append(group_indices, idx)
            test_rank_arr.append(group_indices)
    # Shape (#samples, #group_size)
    test_rank_array = np.array(test_rank_arr, dtype=np.int64)
    all_features, all_labels = get_feature_label(df)
    # Shape (#samples, #group_size, #features)
    rank_group_features = np.take(all_features, test_rank_array, axis=0)
    # Shape (#samples, #group_size)
    rank_group_labels = np.take(all_labels, test_rank_array, axis=0)
    return train_df, test_df, rank_group_features, rank_group_labels


def get_data(data_path, thrpt_threshold=0):
    """Load data from the data path

    Parameters
    ----------
    data_path
        The data path
    thrpt_threshold
        -1 --> adaptive valid thrpt
        0 --> no adaptive valid thrpt

    Returns
    -------
    df
        The DataFrame
    """
    if thrpt_threshold == -1:
        invalid_thd = analyze_valid_threshold(data_path)
    else:
        invalid_thd = 0
    df = read_pd(data_path)
    # Pre-filter the invalid through-puts.
    # For these throughputs, we can directly obtain the result from the ValidNet
    logging.info('Invalid throughput is set to %.1f GFLOP/s', invalid_thd)
    df = df[df['thrpt'] >= invalid_thd]
    assert df.shape[0] > 0

    used_keys = []
    not_used_keys = []
    for key in df.keys():
        if key == 'thrpt':
            used_keys.append(key)
            continue
        if df[key].to_numpy().std() == 0:
            not_used_keys.append(key)
            continue
        used_keys.append(key)
    logging.info('Original keys=%s, Not used keys=%s', list(df.keys()),
                 not_used_keys)
    df = df[used_keys]
    return df, used_keys


def get_feature_label(df):
    feature_keys = [ele for ele in df.keys() if ele != 'thrpt']
    features = df[feature_keys].to_numpy()
    labels = df['thrpt'].to_numpy()
    return features, labels


class CatRegressor:
    def __init__(self, model=None):
        self.model = model

    def fit(self, train_df, valid_df=None, train_dir='.', seed=123):
        if self.model is None:
            params = {
                'loss_function': 'RMSE',
                'task_type': 'GPU',
                'iterations': 10000,
                'verbose': True,
                'train_dir': train_dir,
                'random_seed': seed
            }
            self.model = catboost.CatBoost(params)
            init_model = None
        else:
            incremental_learning = True
            init_model = self.model
        train_features, train_labels = get_feature_label(train_df)
        train_pool = catboost.Pool(data=train_features,
                                   label=train_labels)
        if valid_df is not None:
            valid_features, valid_labels = get_feature_label(valid_df)
            dev_pool = catboost.Pool(data=valid_features,
                                     label=valid_labels)
        else:
            dev_pool = None
        self.model.fit(train_pool, eval_set=dev_pool, init_model=init_model)

    @classmethod
    def load(cls, path):
        try:
            model = catboost.CatBoost().load_model(path)
            return cls(model=model)
        except NameError:  # CatBoost is unavailable. Try to load Python model.
            pass

    def save(self, out_dir):
        self.model.save_model(os.path.join(out_dir, 'cat_regression.cbm'))
        self.model.save_model(os.path.join(out_dir, 'cat_regression'), format='python')

    def predict(self, features):
        features_shape = features.shape
        preds = self.model.predict(features.reshape((-1, features_shape[-1])))
        preds = preds.reshape(features_shape[:-1])
        preds = np.maximum(preds, 0)
        return preds

    def evaluate(self, features, labels, mode='regression'):
        preds = self.predict(features)
        if mode == 'regression':
            rmse = np.sqrt(np.mean(np.square(preds - labels)))
            mae = np.mean(np.abs(preds - labels))
            return {'rmse': rmse, 'mae': mae}
        elif mode == 'ranking':
            # We calculate two things, the NDCG score and the MRR score.
            ndcg_val = ndcg_score(y_true=labels, y_score=preds)
            ranks = np.argsort(-preds, axis=-1) + 1
            true_max_indices = np.argmax(labels, axis=-1)
            rank_of_max = ranks[np.arange(len(true_max_indices)), true_max_indices]
            mrr = np.mean(1.0 / rank_of_max)
            return {'ndcg': ndcg_val, 'mrr': mrr, 'rank_of_top': 1 / mrr}
        else:
            raise NotImplementedError


class CatRanker:
    def __init__(self, model=None):
        self.model = model

    def fit(self, train_df, valid_ranking_features, valid_ranking_labels, train_dir, seed):
        params = {
            'loss_function': 'YetiRank',
            'custom_metric': ['NDCG', 'AverageGain:top=10'],
            'task_type': 'GPU',
            'iterations': 2000,
            'verbose': True,
            'train_dir': train_dir,
            'random_seed': seed
        }
        self.model = catboost.CatBoost(params)
        train_features, train_labels = get_feature_label(train_df)
        train_pool = catboost.Pool(data=train_features,
                                   label=train_labels)
        dev_pool = catboost.Pool(data=valid_features,
                                 label=valid_labels)
        self.model.fit(train_pool, eval_set=dev_pool)
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Performance Model')
    parser.add_argument('--seed',
                        type=int,
                        default=100,
                        help='Seed for the training.')
    parser.add_argument('--data_prefix',
                        type=str,
                        default=None,
                        help='Prefix of the training/validation/testing dataset.')
    parser.add_argument('--out_dir',
                        type=str,
                        default='thrpt_model_out',
                        help='output path of the throughput model.')
    parser.add_argument('--split_test', action='store_true',
                        help='When turned on, we will try to split the data into training, '
                             'and testing.')
    split_args = parser.add_argument_group('data split arguments')
    split_args.add_argument('--dataset',
                            type=str,
                            default=None,
                            help='path to the input csv file.')
    split_args.add_argument('--save_used_keys', action='store_true',
                            help='Store the used keys.')
    split_args.add_argument('--used_key_path', default=None,
                            help='Path of the used key.')
    split_args.add_argument('--split_train_name', default=None,
                            help='Name of the training split.')
    split_args.add_argument('--split_test_name', default=None,
                            help='Name of the testing split.')
    split_args.add_argument('--split_rank_test_name', default=None,
                            help='Name of the rank test model.')
    split_args.add_argument('--split_test_ratio', default=0.1,
                            help='Ratio of the test set in the split.')
    split_args.add_argument('--split_top_ratio', default=0.05,
                            help='Ratio of the top samples that will be split to the test set.')
    split_args.add_argument('--split_rank_group_size', default=10,
                            help='Size of each rank group.')
    split_args.add_argument('--split_rank_K', default=10,
                            help='K of each rank group.')
    parser.add_argument('--algo',
                        choices=['cat_regression',
                                 'cat_ranking'
                                 'nn_regression',
                                 'nn_ranking'],
                        default='cat_regression',
                        help='The algorithm to use.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.split_test:
        logging_config(args.out_dir, 'split_data')
        df, used_keys = get_data(args.dataset)
        train_df, test_df, test_rank_group_features, test_rank_group_labels =\
            split_train_test_df(df,
                                args.seed,
                                args.split_test_ratio,
                                args.split_top_ratio,
                                args.split_rank_group_size,
                                args.split_rank_K)
        logging.info('Generate train data to {}, test data to {}, test rank data to {}'
                     .format(args.split_train_name,
                             args.split_test_name,
                             args.split_rank_test_name))
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        train_df.to_parquet(args.split_train_name)
        test_df.to_parquet(args.split_test_name)
        np.savez(args.split_rank_test_name,
                 rank_features=test_rank_group_features,
                 rank_labels=test_rank_group_labels)
        logging.info('  #Train = {}, #Test = {}, #Ranking Test Groups = {}'
                     .format(len(train_df),
                             len(test_df),
                             len(test_rank_group_features)))
        if args.save_used_keys:
            with open(args.used_key_path, 'w') as of:
                json.dump(used_keys, of)
    else:
        logging_config(args.out_dir, 'train')
        train_df = read_pd(args.data_prefix + '.train.pq')
        valid_df = read_pd(args.data_prefix + '.valid.pq')
        test_df = read_pd(args.data_prefix + '.test.pq')
        rank_valid = np.load(args.data_prefix + '.rank_valid.npz')
        rank_test = np.load(args.data_prefix + '.rank_test.npz')
        with open(args.data_prefix + '.used_key.json', 'r') as in_f:
            used_key = json.load(in_f)
        train_df = train_df[used_key]
        valid_df = valid_df[used_key]
        test_df = test_df[used_key]
        if args.algo == 'cat_regression':
            model = CatRegressor()
            model.fit(train_df, valid_df, train_dir=args.out_dir, seed=args.seed)
            model.save(args.out_dir)
            valid_features, valid_labels = get_feature_label(valid_df)
            test_features, test_labels = get_feature_label(test_df)
            valid_score = model.evaluate(valid_features, valid_labels, 'regression')
            test_score = model.evaluate(test_features, test_labels, 'regression')
            valid_ranking_score = model.evaluate(rank_valid['rank_features'],
                                                 rank_valid['rank_labels'],
                                                 'ranking')
            test_ranking_score = model.evaluate(rank_test['rank_features'],
                                                rank_test['rank_labels'],
                                                'ranking')
            valid_score.update(valid_ranking_score)
            test_score.update(test_ranking_score)
            logging.info('Valid Score={}'.format(valid_score))
            logging.info('Test Score={}'.format(test_score))
            with open(os.path.join(args.out_dir, 'valid_scores.json'), 'w') as out_f:
                json.dump(valid_score, out_f)
            with open(os.path.join(args.out_dir, 'test_scores.json'), 'w') as out_f:
                json.dump(test_score, out_f)


if __name__ == "__main__":
    main()
