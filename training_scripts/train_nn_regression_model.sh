set -ex

neg_mult=$1
iter_mult=$2
units=${3:-512}
num_layers=${4:-3}
dropout=${5:-0.1}
use_gate=${6:-1}
num_gpus=${7:-1}
cuda_device=${8:-0}
task=$9


export CUDA_VISIBLE_DEVICES=$((${cuda_device} % ${num_gpus}))

TUNING_DATASET=../tuning_dataset
data_prefix=../split_tuning_dataset/$task
MODEL_DIR=../model_results/nn_regression_${neg_mult}_${iter_mult}
mkdir -p ${MODEL_DIR}
python3 -m perf_model.thrpt_model_new \
    --algo nn \
    --rank_loss_type no_rank \
    --data_prefix ${data_prefix} \
    --iter_mult ${iter_mult} \
    --neg_mult ${neg_mult} \
    --dropout ${dropout} \
    --use_gate ${use_gate} \
    --out_dir ${MODEL_DIR}/$task
cp ${TUNING_DATASET}/$task.meta ${MODEL_DIR}/$task/feature.meta
