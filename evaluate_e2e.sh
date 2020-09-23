MODEL_NAME=cat_regression
MODEL_PATH=./model_results/${MODEL_NAME}/gcv_t4_csv
MODEL_TYPE=cat_regression
MODELS=(
"InceptionV3"
"MobileNet1.0"
"MobileNetV2_1.0"
"ResNet18_v2"
"ResNet50_v2"
"SqueezeNet1.0"
"SqueezeNet1.1"
"VGG19"
)
TARGET="cuda -model=t4"
for network in ${MODELS}
do
  python3 app/main.py --list-net ${MODEL_PATH} \
                    --model_type ${MODEL_TYPE} \
                    --target $TARGET --gcv ${network} 2>&1 | tee -a ${MODEL_NAME}.txt
done;
