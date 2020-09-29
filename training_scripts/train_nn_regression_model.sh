set -ex

neg_mult=$1
iter_mult=$2
task=$3

TUNING_DATASET=../tuning_dataset
data_prefix=../split_tuning_dataset/$task
MODEL_DIR=../model_results/nn_${loss_type}_${rank_lambda}_${iter_mult}
mkdir -p ${MODEL_DIR}
python3 -m perf_model.thrpt_model_new \
    --algo nn \
    --rank_loss_type no_rank \
    --data_prefix ${data_prefix} \
    --iter_mult ${iter_mult} \
    --neg_mult ${neg_mult} \
    --out_dir ${MODEL_DIR}/$task
cp ${TUNING_DATASET}/$task.meta ${MODEL_DIR}/$task/feature.meta
