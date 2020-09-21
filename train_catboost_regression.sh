set -ex

mkdir -p model_results
for train_file in `ls split_tuning_dataset/*/*.train.pq`;
do
  data_prefix=${train_file:0:-9}
  folder_prefix=${split_tuning_dataset:21:-9}
  python3 -m perf_model.thrpt_model_new \
      --algo cat_regression \
      --data_prefix ${data_prefix} \
      --out_dir model_results/${folder_prefix}/cat_regression
done;
