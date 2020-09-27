set -ex

n_parallel=8
measure_top_n=32
TARGET="llvm -mcpu=skylake-avx512"
FOLDER="gcv_skylake_csv"
MODELS=(
"InceptionV3"
"MobileNet1.0"
"MobileNetV2_1.0"
"ResNet18_v2"
"ResNet50_v2"
"SqueezeNet1.0"
"SqueezeNet1.1"
"VGG19"
"VGG19_bn"
"ResNet50_v1"
"ResNet18_v1"
)

for MODEL_NAME in nn_0.0_40 nn_0.0_80 nn_1.0_40 nn_1.0_80 nn_1.0_40_hinge nn_1.0_80_hinge;
do
  MODEL_PATH=../model_results/${MODEL_NAME}/${FOLDER}

  OUT_DIR=${MODEL_NAME}_e2e_t4_npara${n_parallel}_ntop${measure_top_n}
  mkdir -p ${OUT_DIR}
  for network in ${MODELS[@]}
  do
    python3 ../app/main.py --list-net ${MODEL_PATH} \
                      --model_type ${MODEL_TYPE} \
                      --n-parallel ${n_parallel} \
                      --measure-top-n ${measure_top_n} \
                      --graph \
                      --target "${TARGET}" --gcv ${network} 2>&1 | tee -a ${OUT_DIR}/${network}.txt
  done;
done;