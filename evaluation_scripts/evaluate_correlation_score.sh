for model in cat_regression_split0.3 cat_regression_split0.5 cat_regression_split0.7 cat_regression_split1
do
  python3 evaluate.py --eval_correlation --dir_path model_results/${model} --model_type cat_regression --correlation_out_name ${model}_corr
done

