# pylint: disable=missing-docstring, invalid-name, ungrouped-imports
# pylint: disable=unsupported-assignment-operation, no-member
import argparse
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import torch as th
import numpy as np
import random
import pandas as pd
import tqdm
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from util import analyze_valid_threshold, logging_config


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
    test_rank_df
        The dataframe that is used to verify the ranking model.
        (#Test * K, group_size)
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
        perm = rng.permutation(len(df))
        train_indices = perm[:train_num]
        test_indices = perm[train_num:]
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    # Get ranking dataframe, for each sample in the test set, randomly sample
    # group_size - 1 elements
    test_rank_df = []
    for i, idx in enumerate(test_indices):
        for _ in range(K):
            group_indices = (rng.choice(num_samples - 1, group_size - 1, True) + idx + 1) % num_samples
            group_indices = np.append(group_indices, idx)
            test_rank_df.append(group_indices)
    test_rank_df = df.DataFrame(test_rank_df)
    return train_df, test_df, test_rank_df


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
    df = pd.read_csv(data_path)
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
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Performance Model')
    parser.add_argument('--seed',
                        type=int,
                        default=100,
                        help='Seed for the training.')
    parser.add_argument('--train_dataset',
                        type=str,
                        default=None,
                        help='path to the training csv file.')
    parser.add_argument('--test_dataset',
                        type=str,
                        default=None,
                        help='path to the test csv file.')
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
                            required=True,
                            help='path to the input csv file.')
    split_args.add_argument('--split_train_name', default=None,
                            help='Name of the training split.')
    split_args.add_argument('--split_test_name', default=None,
                            help='Name of the testing split.')
    split_args.add_argument('--split_rank_test_name', default=None,
                            help='Name of the rank test model.')
    split_args.add_argument('--split_test_ratio', default=0.1,
                            help='Ratio of the test set in the split.')
    split_args.add_argument('--split_top_ratio', default=0.2,
                            help='Ratio of the top samples that will be split to the test set.')
    split_args.add_argument('--split_rank_group_size', default=10,
                            help='Size of each rank group.')
    split_args.add_argument('--split_rank_K', default=4,
                            help='K of each rank group.')
    parser.add_argument('--algo',
                        choices=['cat', 'nn'],
                        default='cat',
                        help='The algorithm to use.')
    parser.add_argument('--problem_type',
                        choices=['regression', 'ranking'],
                        default='ranking',
                        help='The problem type')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging_config()
    set_seed(args.seed)
    logging_config(args.out_dir, 'thrpt_model')

    if args.split_test:
        df = get_data(args.dataset)
        train_df, test_df, test_rank_df = split_train_test_df(df,
                                                              args.seed,
                                                              args.split_test_ratio,
                                                              args.split_top_ratio,
                                                              args.split_rank_group_size,
                                                              args.split_rank_K)
        train_df.to_csv(args.split_train_name)
        test_df.to_csv(args.split_test_name)
        test_rank_df.to_csv(args.split_rank_test_name)
    else:
        pass


if __name__ == "__main__":
    main()
