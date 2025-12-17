# Hyperparameter Contribution Report

- Files parsed: 111
- Rows used (target not-null): 109

## Target

- target: `final_train_perplexity`  (lower is better)

## CV sanity check

- CV MAE (mean ± std): 23.3893 ± 7.77796

## Best / Worst run

- Best: `NGRC_LM(40.76M_d512_lag10_poly3_rank512_lr0.0005_bs200_seq256_20251217-005340`  (final_train_perplexity=207.60992002029425)
- Worst: `NGRC_LM(26.95M_d64_lag256_poly1_rank512_lr0.0005_bs200_seq256_20251214-101708` (final_train_perplexity=814.4645153364638)

## Top 15 important hyperparameters (Permutation Importance)

| param               |   importance_mean |   importance_std |
|:--------------------|------------------:|-----------------:|
| hp.ngrc_d_model     |      86.1166      |      14.1643     |
| hp.ngrc_poly_degree |      42.0658      |       2.69549    |
| hp.ngrc_lag         |      15.8592      |       3.98536    |
| hp.learning_rate    |      -5.9508e-16  |       4.9797e-15 |
| hp.local_batch_size |      -1.94511e-15 |       7.0623e-15 |

## Numeric params: Spearman correlation hint (direction)

- rho < 0 : 値が大きいほど target が下がりやすい（改善しやすい傾向）
- rho > 0 : 値が大きいほど target が上がりやすい（悪化しやすい傾向）

| param               |   spearman_rho |     p_value |   n |     min |     max |
|:--------------------|---------------:|------------:|----:|--------:|--------:|
| hp.ngrc_d_model     |      -0.763858 | 4.47725e-22 | 109 | 64      | 512     |
| hp.ngrc_lag         |       0.505911 | 2.0049e-08  | 109 |  5      | 256     |
| hp.ngrc_poly_degree |      -0.31059  | 0.00113014  | 107 |  1      |   7     |
| hp.local_batch_size |      -0.117319 | 0.224393    | 109 | 32      | 200     |
| hp.learning_rate    |       0.117319 | 0.224393    | 109 |  0.0005 |   0.001 |
