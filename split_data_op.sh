set -ex
OUT_DIR=split_tuning_dataset_op
mkdir -p ${OUT_DIR}

for fold in `cat tasks.txt`;
do
    mkdir -p ${OUT_DIR}/${fold}
    for fname in `ls tuning_dataset/$fold/*.csv`;
    do
        prefix_name=${fname:0:-4}
        python3 -m perf_model.thrpt_model_new  \
                --split_test_op_level \
                --dataset $fname \
                --split_train_name ${prefix_name}.train.pq \
                --split_test_name ${prefix_name}.test.pq \
                --split_test_ratio 0.1 \
                --seed 123
        mv ${prefix_name}.train.pq ${OUT_DIR}/${fold}
        mv ${prefix_name}.test.pq ${OUT_DIR}/${fold}
    done;
done;
