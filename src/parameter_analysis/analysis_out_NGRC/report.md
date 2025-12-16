# Hyperparameter Contribution Report

- Files parsed: 48
- Rows used (target not-null): 46

## Target

- target: `final_train_perplexity`  (lower is better)

## CV sanity check

- CV MAE (mean ± std): 38.8542 ± 17.605

## Best / Worst run

- Best: `NGRC_LM(45.48M_d512_lag16_poly3_rank512_lr0.0005_bs200_seq256_20251214-113932`  (final_train_perplexity=212.3648808057814)
- Worst: `NGRC_LM(26.95M_d64_lag256_poly1_rank512_lr0.0005_bs200_seq256_20251214-101708` (final_train_perplexity=814.4645153364638)

## Top 15 important hyperparameters (Permutation Importance)

| param               |   importance_mean |   importance_std |
|:--------------------|------------------:|-----------------:|
| hp.ngrc_d_model     |      71.352       |     12.696       |
| hp.ngrc_poly_degree |      40.8802      |      5.35301     |
| hp.ngrc_lag         |      18.373       |      3.90808     |
| hp.local_batch_size |      -1.50768e-14 |      1.82263e-14 |
| hp.learning_rate    |      -1.52323e-14 |      1.54955e-14 |

## Numeric params: Spearman correlation hint (direction)

- rho < 0 : 値が大きいほど target が下がりやすい（改善しやすい傾向）
- rho > 0 : 値が大きいほど target が上がりやすい（悪化しやすい傾向）

| param               |   spearman_rho |     p_value |   n |     min |     max |
|:--------------------|---------------:|------------:|----:|--------:|--------:|
| hp.ngrc_d_model     |     -0.679286  | 2.10191e-07 |  46 | 64      | 512     |
| hp.ngrc_lag         |      0.442599  | 0.00206973  |  46 | 16      | 256     |
| hp.ngrc_poly_degree |     -0.380945  | 0.0107366   |  44 |  1      |   7     |
| hp.local_batch_size |     -0.0481776 | 0.750524    |  46 | 32      | 200     |
| hp.learning_rate    |      0.0481776 | 0.750524    |  46 |  0.0005 |   0.001 |
