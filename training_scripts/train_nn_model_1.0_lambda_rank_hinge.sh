set -ex

rank_lambda=1.0
source ./tasks.sh

N=8
i=0
for iter_mult in 120;
do
  for task in ${TASKS[@]};
  do
    ((i=i%N))
    ((i++==0)) && wait
    echo $i
    ( data_prefix=split_tuning_dataset/$task
      MODEL_DIR=model_results/nn_${rank_lambda}_${iter_mult}_hinge
      mkdir -p ${MODEL_DIR}
      python3 -m perf_model.thrpt_model_new \
          --algo nn \
          --data_prefix ${data_prefix} \
          --rank_lambda ${rank_lambda} \
          --rank_loss_type lambda_rank_hinge \
          --iter_mult ${iter_mult} \
          --out_dir ${MODEL_DIR}/$task ) &
  done;
done;
