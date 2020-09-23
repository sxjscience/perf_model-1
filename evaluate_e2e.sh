set -ex

MODEL_PATH=./model_results/cat_regression/gcv_t4_csv
MODEL_TYPE=cat_regression
OUT_DIR=cat_regression_e2e_t4
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
TARGET="cuda -model=t4"
mkdir -p ${OUT_DIR}
for network in ${MODELS[@]}
do
  python3 app/main.py --list-net ${MODEL_PATH} \
                    --model_type ${MODEL_TYPE} \
                    --target "${TARGET}" --gcv ${network} 2>&1 | tee -a ${OUT_DIR}/${network}.txt
done;
