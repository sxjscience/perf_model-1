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


### Run NoRank Ablation

```
cat tasks.txt | parallel -j 8 bash train_nn_regression_model.sh -1 1000 512 3 0.1 0
cat tasks.txt | parallel -j 8 bash train_nn_regression_model.sh -1 1000 512 3 0.1 1
cat tasks.txt | parallel -j 8 bash train_nn_regression_model.sh 1 1000 512 3 0.1 1
```