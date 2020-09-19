set -ex

for fold in `ls tuning_dataset`;
do
    for fname in `ls tuning_dataset/$fold/*.csv`;
    do
        python3 perf_model/thrpt_model_new.py  \
                --split_test \
                --dataset $fname \
                --split_train_name $fname.train \
                --split_test_name $fname.test \
                --split_rank_test_name $fname.rank_test \
                --seed 123
        python3 perf_model/thrpt_model_new.py  \
                --split_test \
                --dataset $fname.train \
                --split_train_name $fname.train \
                --split_test_name $fname.valid \
                --split_rank_test_name $fname.rank_valid \
                --seed 123
    done;
done;
