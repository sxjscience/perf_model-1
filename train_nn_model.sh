set -ex

rank_lambda=5.0
iter_mult=500

mkdir -p model_results/cat_regression
for train_file in `ls split_tuning_dataset/*/*.train.pq`;
do
  data_prefix=${train_file:0:-9}
  folder_prefix=${train_file:21:-9}
  python3 -m perf_model.thrpt_model_new \
      --algo nn \
      --data_prefix ${data_prefix} \
      --rank_lambda ${rank_lambda} \
      --iter_mult ${iter_mult} \
      --out_dir model_results/nn_${rank_lambda}_${iter_mult}/${folder_prefix}
done;
