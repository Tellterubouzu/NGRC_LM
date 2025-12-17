# Hyperparameter Contribution Report

- Files parsed: 30
- Rows used (target not-null): 30

## Target

- target: `final_train_perplexity`  (lower is better)

## CV sanity check

- CV MAE (mean ± std): 213.011 ± 49.3208

## Best / Worst run

- Best: `multilayer_NGRC(56.11M_d256_lag16_poly3_layer6_rank512_lr0.0005_bs200_seq256_20251216-231404`  (final_train_perplexity=317.1571862671759)
- Worst: `multilayer_NGRC(61.09M_d128_lag32_poly2_layer12_rank512_lr0.0005_bs200_seq256_20251217-003922` (final_train_perplexity=1746.6068811371654)

## Top 15 important hyperparameters (Permutation Importance)

| param               |   importance_mean |   importance_std |
|:--------------------|------------------:|-----------------:|
| hp.ngrc_num_layers  |         286.62    |         53.129   |
| hp.ngrc_lag         |          15.5366  |         22.6694  |
| hp.ngrc_d_model     |           5.57452 |          6.0161  |
| hp.ngrc_poly_degree |           2.16829 |          1.62675 |

## Numeric params: Spearman correlation hint (direction)

- rho < 0 : 値が大きいほど target が下がりやすい（改善しやすい傾向）
- rho > 0 : 値が大きいほど target が上がりやすい（悪化しやすい傾向）

| param               |   spearman_rho |     p_value |   n |   min |   max |
|:--------------------|---------------:|------------:|----:|------:|------:|
| hp.ngrc_num_layers  |       0.723389 | 6.28464e-06 |  30 |     6 |    18 |
| hp.ngrc_lag         |       0.258231 | 0.168268    |  30 |    16 |    96 |
| hp.ngrc_d_model     |       0.190473 | 0.313359    |  30 |    64 |   512 |
| hp.ngrc_poly_degree |       0.123831 | 0.514434    |  30 |     1 |     3 |
