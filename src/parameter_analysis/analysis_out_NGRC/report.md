# Hyperparameter Contribution Report

- Files parsed: 146
- Rows used (target not-null): 144

## Target

- target: `final_train_perplexity`  (lower is better)

## CV sanity check

- CV MAE (mean ± std): 18.9005 ± 3.3207

## Best / Worst run

- Best: `NGRC_LM(113.51M_d2048_lag10_poly3_rank512_lr0.0005_bs150_seq256_20251217-142020`  (final_train_perplexity=114.84103526818075)
- Worst: `NGRC_LM(26.95M_d64_lag256_poly1_rank512_lr0.0005_bs200_seq256_20251214-101708` (final_train_perplexity=814.4645153364638)

## Top 15 important hyperparameters (Permutation Importance)

| param               |   importance_mean |   importance_std |
|:--------------------|------------------:|-----------------:|
| hp.ngrc_d_model     |      96.5798      |     11.4563      |
| hp.ngrc_poly_degree |      32.5352      |      9.36126     |
| hp.ngrc_lag         |      13.5688      |      2.58371     |
| hp.local_batch_size |       1.06455     |      0.702733    |
| hp.learning_rate    |      -1.08269e-14 |      1.38122e-14 |

## Numeric params: Spearman correlation hint (direction)

- rho < 0 : 値が大きいほど target が下がりやすい（改善しやすい傾向）
- rho > 0 : 値が大きいほど target が上がりやすい（悪化しやすい傾向）

| param               |   spearman_rho |     p_value |   n |     min |      max |
|:--------------------|---------------:|------------:|----:|--------:|---------:|
| hp.ngrc_d_model     |     -0.847057  | 8.27208e-41 | 144 | 64      | 4096     |
| hp.ngrc_lag         |      0.556735  | 4.29763e-13 | 144 |  5      |  256     |
| hp.ngrc_poly_degree |     -0.263976  | 0.00150183  | 142 |  1      |    7     |
| hp.learning_rate    |      0.127048  | 0.129153    | 144 |  0.0005 |    0.001 |
| hp.local_batch_size |     -0.0373171 | 0.657001    | 144 | 32      |  400     |
