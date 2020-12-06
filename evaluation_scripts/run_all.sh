#bash evaluate_e2e.sh g4 cat_regression_op_5000_split1 cat_regression
#bash evaluate_e2e.sh g4 cat_ranking_op_5000_split1 cat_ranking

for model in nn_regression_op_new_split1_-1_1000_512_3_0.1_1 \
             nn_regression_op_new_split1_-1_1000_512_3_0.1_0 \
             nn_regression_op_new_split0.7_-1_1000_512_3_0.1_1 \
             nn_regression_op_new_split0.5_-1_1000_512_3_0.1_1 \
             nn_regression_op_new_split0.3_-1_1000_512_3_0.1_1
do
    for seed in 123
    do
        for K in 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;
