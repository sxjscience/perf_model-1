# Run Training Scripts

```
# Run with LambdaRank-Hinge 
cat tasks.txt | parallel -j 8 bash train_nn_model.sh lambda_rank_hinge 1.0 120
# Run with LambdaRank
cat tasks.txt | parallel -j 8 bash train_nn_model.sh lambda_rank 1.0 120
# Run with NoRank
cat tasks.txt | parallel -j 8 bash train_nn_model.sh lambda_rank 0.0 120
```