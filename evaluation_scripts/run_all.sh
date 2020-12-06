bash evaluate_e2e.sh g4 cat_regression_op_5000_split1 cat_regression
bash evaluate_e2e.sh g4 cat_ranking_op_5000_split1 cat_ranking

# Evaluate NN + Gate

for seed in 123
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_1
    do
        for K in 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;


for seed in 123
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_1
    do
        for K in 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;


for seed in 123
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_0
    do
        for K in 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;


# 0.3
for seed in 123
do
    for model in nn_regression_split0.3_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;

# 0.5
for seed in 123
do
    for model in nn_regression_split0.5_-1_1000_512_3_0.1_1
    do
        for K in 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;

# 0.7
for seed in 123
do
    for model in nn_regression_split0.7_-1_1000_512_3_0.1_1
    do
        for K in 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;
