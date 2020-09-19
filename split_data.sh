set -ex

mkdir split_tuning_dataset
for fold in `ls tuning_dataset`;
do
    mkdir split_tuning_dataset/$fold
    for fname in `ls tuning_dataset/$fold/*.csv`;
    do
        python3 perf_model/thrpt_model_new.py  \
                --split_test \
                --dataset tuning_dataset/$fold/$fname \
                --split_train_name split_tuning_dataset/$fold/$fname.train \
                --split_test_name split_tuning_dataset/$fold/$fname.test \
                --split_rank_test_name split_tuning_dataset/$fold/$fname.rank_test
    done;
done;
