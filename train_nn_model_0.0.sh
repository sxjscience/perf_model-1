set -ex

rank_lambda=0.0
iter_mult=200

MODEL_DIR=model_results/nn_${rank_lambda}_${iter_mult}
mkdir -p ${MODEL_DIR}
for train_file in `ls split_tuning_dataset/*/*.train.pq`;
do
  data_prefix=${train_file:0:-9}
  folder_prefix=${train_file:21:-9}
  python3 -m perf_model.thrpt_model_new \
      --algo nn \
      --data_prefix ${data_prefix} \
      --rank_lambda ${rank_lambda} \
      --iter_mult ${iter_mult} \
      --out_dir ${MODEL_DIR}/${folder_prefix}
done;
