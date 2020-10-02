for model in cat_regression_split0.3 cat_regression_split0.5 cat_regression_split0.7 cat_regression_split1
do
  python3 evaluate.py --eval_correlation --dir_path model_results/${model} --model_type cat_regression --correlation_out_name ${model}_corr
done


for model in nn_regression_split0.3_-1_1000_512_3_0.1_0 \
             nn_regression_split0.3_-1_1000_512_3_0.1_1 \
             nn_regression_split0.5_-1_1000_512_3_0.1_0 \
             nn_regression_split0.5_-1_1000_512_3_0.1_1 \
             nn_regression_split0.7_-1_1000_512_3_0.1_0 \
             nn_regression_split0.7_-1_1000_512_3_0.1_1 \
             nn_regression_split1_-1_1000_512_3_0.1_0 \
             nn_regression_split1_-1_1000_512_3_0.1_1
do
  python3 evaluate.py --eval_correlation --dir_path model_results/${model} --model_type nn --correlation_out_name ${model}_corr
done
