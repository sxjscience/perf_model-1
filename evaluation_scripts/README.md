# Run End-to-end Performance Tuning with the Trained Performance Model

## G4
```bash
# Evaluate Catboost Regression on G4
bash evaluate_e2e.sh g4 cat_regression_split1 cat_regression
bash evaluate_e2e.sh g4 cat_regression_split0.7 cat_regression
bash evaluate_e2e.sh g4 cat_regression_split0.5 cat_regression

# Evaluate Catboost Ranking on G4
bash evaluate_e2e.sh g4 cat_ranking cat_ranking

# Evaluate neural network
networks=(
nn_regression_split1_-1_1000_512_3_0.1_0
nn_regression_split1_-1_1000_512_3_0.1_1
nn_regression_split0.7_-1_1000_512_3_0.1_0
nn_regression_split0.7_-1_1000_512_3_0.1_1
nn_regression_split0.5_-1_1000_512_3_0.1_0
nn_regression_split0.5_-1_1000_512_3_0.1_1 
nn_regression_split0.3_-1_1000_512_3_0.1_0
nn_regression_split0.3_-1_1000_512_3_0.1_1
)

for model in ${networks[@]} 
do
    bash evaluate_e2e.sh g4 ${model} nn
done;
```

## C5
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh c5 cat_regression_split1 cat_regression
bash evaluate_e2e.sh c5 cat_regression_split0.7 cat_regression
bash evaluate_e2e.sh c5 cat_regression_split0.5 cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh c5 cat_ranking cat_ranking
```

## C4
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh c4 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh c4 cat_ranking cat_ranking

bash evaluate_e2e.sh c4 nn_regression_split1_-1_1000_512_3_0.1_0 nn 8
bash evaluate_e2e.sh c4 nn_regression_split1_-1_1000_512_3_0.1_1 nn 8
```

## P3
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh p3 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh p3 cat_ranking cat_ranking

# NN
bash evaluate_e2e.sh p3 nn_regression_split1_-1_1000_512_3_0.1_0 nn 2
bash evaluate_e2e.sh p3 nn_regression_split1_-1_1000_512_3_0.1_1 nn 2
```

## M6

```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh m6 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh m6 cat_ranking cat_ranking
```
