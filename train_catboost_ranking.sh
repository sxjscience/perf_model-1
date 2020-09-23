set -ex

mkdir -p model_results/cat_ranking
for train_file in `ls split_tuning_dataset/*/*.train.pq`;
do
  data_prefix=${train_file:0:-9}
  folder_prefix=${train_file:21:-9}
  python3 -m perf_model.thrpt_model_new \
      --algo cat_ranking \
      --data_prefix ${data_prefix} \
      --out_dir model_results/cat_ranking/${folder_prefix}
done;
