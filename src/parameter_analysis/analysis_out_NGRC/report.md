# Hyperparameter Contribution Report

- Files parsed: 87
- Rows used (target not-null): 85

## Target

- target: `final_train_perplexity`  (lower is better)

## CV sanity check

- CV MAE (mean ± std): 27.5626 ± 9.27203

## Best / Worst run

- Best: `NGRC_LM(45.48M_d512_lag16_poly3_rank512_lr0.0005_bs200_seq256_20251214-113932`  (final_train_perplexity=212.3648808057814)
- Worst: `NGRC_LM(26.95M_d64_lag256_poly1_rank512_lr0.0005_bs200_seq256_20251214-101708` (final_train_perplexity=814.4645153364638)

## Top 15 important hyperparameters (Permutation Importance)

| param               |   importance_mean |   importance_std |
|:--------------------|------------------:|-----------------:|
| hp.ngrc_d_model     |      77.082       |     12.6176      |
| hp.ngrc_poly_degree |      44.9719      |      7.52365     |
| hp.ngrc_lag         |      14.0526      |      3.00505     |
| hp.local_batch_size |       2.75335e-15 |      2.38458e-14 |
| hp.learning_rate    |       2.6823e-15  |      2.14876e-14 |

## Numeric params: Spearman correlation hint (direction)

- rho < 0 : 値が大きいほど target が下がりやすい（改善しやすい傾向）
- rho > 0 : 値が大きいほど target が上がりやすい（悪化しやすい傾向）

| param               |   spearman_rho |     p_value |   n |     min |     max |
|:--------------------|---------------:|------------:|----:|--------:|--------:|
| hp.ngrc_d_model     |     -0.703073  | 6.31147e-14 |  85 | 64      | 512     |
| hp.ngrc_lag         |      0.418505  | 6.72319e-05 |  85 | 16      | 256     |
| hp.ngrc_poly_degree |     -0.34934   | 0.00120794  |  83 |  1      |   7     |
| hp.local_batch_size |     -0.0949006 | 0.387622    |  85 | 32      | 200     |
| hp.learning_rate    |      0.0949006 | 0.387622    |  85 |  0.0005 |   0.001 |
