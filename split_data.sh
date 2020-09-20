set -ex
mkdir -p split_tuning_dataset

for fold in `ls tuning_dataset`;
do
    mkdir split_tuning_dataset/${fold}
    for fname in `ls tuning_dataset/$fold/*.csv`;
    do
        prefix_name=${fname:0:-4}
        python3 perf_model/thrpt_model_new.py  \
                --split_test \
                --save_used_keys \
                --used_key_path ${prefix_name}.used_key.json \
                --dataset $fname \
                --split_train_name ${prefix_name}.train.csv \
                --split_test_name ${prefix_name}.test.csv \
                --split_rank_test_name ${prefix_name}.rank_test.npy \
                --seed 123
        python3 perf_model/thrpt_model_new.py  \
                --split_test \
                --dataset ${prefix_name}.train.csv \
                --split_train_name ${prefix_name}.train.csv \
                --split_test_name ${prefix_name}.valid.csv \
                --split_rank_test_name ${prefix_name}.rank_valid.npy \
                --seed 123
        mv ${prefix_name}.train.csv split_tuning_dataset/${fold}
        mv ${prefix_name}.valid.csv split_tuning_dataset/${fold}
        mv ${prefix_name}.test.csv split_tuning_dataset/${fold}
        mv ${prefix_name}.rank_valid.npy split_tuning_dataset/${fold}
        mv ${prefix_name}.rank_test.npy split_tuning_dataset/${fold}
    done;
done;
