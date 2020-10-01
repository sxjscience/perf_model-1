# Run Training Scripts

```
# Run with LambdaRank-Hinge 
cat tasks.txt | parallel -j 8 bash train_nn_model.sh lambda_rank_hinge 1.0 120
# Run with LambdaRank
cat tasks.txt | parallel -j 8 bash train_nn_model.sh lambda_rank 1.0 120
# Run with NoRank
cat tasks.txt | parallel -j 8 bash train_nn_model.sh lambda_rank 0.0 120

# Run CatBoost Regression
cat tasks.txt | parallel -j 1 bash train_catboost.sh cat_regression 5000

# Run CatBoost Ranking
cat tasks.txt | parallel -j 1 bash train_catboost.sh cat_ranking 5000
```


### Run Neural Network Performance Model Ablation

Use a p3.16x instance.
```
# Baseline
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh -1 1000 512 3 0.1 0 8
# Baseline + Gate
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh -1 1000 512 3 0.1 1 8
# Baseline + Gate + Balanced
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh 2 1000 512 3 0.1 1 8
```
