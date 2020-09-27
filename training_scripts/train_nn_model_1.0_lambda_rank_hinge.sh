set -ex

rank_lambda=1.0
TASKS=(
"gcv_graviton2_csv/conv2d_NCHWc.x86"
"gcv_graviton2_csv/conv2d_nchw_spatial_pack.arm_cpu"
"gcv_graviton2_csv/conv2d_nchw_winograd.arm_cpu"
"gcv_graviton2_csv/dense_nopack.x86"
"gcv_graviton2_csv/dense_pack.x86"
"gcv_graviton2_csv/depthwise_conv2d_NCHWc.x86"
"gcv_graviton2_csv/depthwise_conv2d_nchw.arm_cpu"

"gcv_skylake_csv/conv2d_NCHWc.x86"
"gcv_skylake_csv/dense_nopack.x86"
"gcv_skylake_csv/dense_pack.x86"
"gcv_skylake_csv/depthwise_conv2d_NCHWc.x86"

#"gcv_t4_csv/conv2d_cudnn.cuda"
"gcv_t4_csv/conv2d_nchw.cuda"
"gcv_t4_csv/conv2d_nchw_winograd.cuda"
"gcv_t4_csv/conv2d_transpose_nchw.cuda"
#"gcv_t4_csv/dense_cublas.cuda"
#"gcv_t4_csv/dense_large_batch.cuda"
#"gcv_t4_csv/dense_small_batch.cuda"
"gcv_t4_csv/dense_tensorcore.cuda"
"gcv_t4_csv/depthwise_conv2d_nchw.cuda"

#"gcv_v100_csv/conv2d_cudnn.cuda"
"gcv_v100_csv/conv2d_nchw.cuda"
"gcv_v100_csv/conv2d_nchw_winograd.cuda"
"gcv_v100_csv/conv2d_transpose_nchw.cuda"
#"gcv_v100_csv/dense_cublas.cuda"
#"gcv_v100_csv/dense_large_batch.cuda"
#"gcv_v100_csv/dense_small_batch.cuda"
"gcv_v100_csv/dense_tensorcore.cuda"
"gcv_v100_csv/depthwise_conv2d_nchw.cuda"
)


N=8
(
for iter_mult in 120;
do
  for task in ${TASKS[@]};
  do
    ((i=i%N)); ((i++==0)) && wait
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
)
