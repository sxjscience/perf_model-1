set -ex

n_parallel=8
measure_top_n=32
instance_type=$1
model_name=$2
model_type=$3

if [ "${instance_type}" -eq "g4" ]; then
  TARGET="cuda -model=t4"
  FOLDER="gcv_t4_csv"
  USE_GRAPH=0
elif [ "${instance_type}" -eq "c4" ]; then
  TARGET="llvm -mcpu=core-avx2"
  FOLDER="gcv_skylake_csv"
  USE_GRAPH=1
elif [ "${instance_type}" -eq "c5" ]; then
  TARGET="llvm -mcpu=skylake-avx512"
  FOLDER="gcv_skylake_csv"
  USE_GRAPH=1
elif [ "${instance_type}" -eq "p3" ]; then
  TARGET="cuda -model=v100"
  FOLDER="gcv_v100_csv"
  USE_GRAPH=0
elif [ "${instance_type}" -eq "m6" ]; then
  TARGET="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+dotprod,+crc,+crypto,+neon"
  FOLDER="gcv_graviton_csv"
  USE_GRAPH=0
else
  echo "${instance_type} not supported!"
  exit
fi

MODEL_PATH=../model_results/${model_name}

OUT_DIR=${model_name}_e2e_{instance_type}_npara${n_parallel}_ntop${measure_top_n}
mkdir -p ${OUT_DIR}
for network in `cat models.txt`
do
  if [ ${USE_GRAPH} -eq 0 ]; then
    python3 ../app/main.py --list-net ${MODEL_PATH} \
                      --model_type ${model_type} \
                      --n-parallel ${n_parallel} \
                      --measure-top-n ${measure_top_n} \
                      --target "${TARGET}" --gcv ${network} 2>&1 | tee -a ${OUT_DIR}/${network}.txt
 else
   python3 ../app/main.py --list-net ${MODEL_PATH} \
                        --model_type ${model_type} \
                        --n-parallel ${n_parallel} \
                        --measure-top-n ${measure_top_n} \
                        --graph \
                        --target "${TARGET}" --gcv ${network} 2>&1 | tee -a ${OUT_DIR}/${network}.txt
 fi
done;
