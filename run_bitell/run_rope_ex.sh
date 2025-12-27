cd ../src


source ~/miniconda3/etc/profile.d/conda.sh
conda activate ais

python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 5e-4 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 1 --ngrc_cross_mode "random" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr5e-4_d2048_poly1_bs32)"

python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 1e-3 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 1 --ngrc_cross_mode "random" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr1e-3_d2048_poly1_bs32)"


python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 5e-4 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 1 --ngrc_cross_mode "none" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr5e-5_d2048_poly1_bs32)"

python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 1e-3 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 1 --ngrc_cross_mode "none" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr1e-3_d2048_poly1_bs32)"

python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 5e-4 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3 --ngrc_cross_mode "random" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr5e-4_d2048_poly3_bs32)"

python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 1e-3 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3 --ngrc_cross_mode "random" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr1e-3_d2048_poly3_bs32)"


python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 5e-4 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3 --ngrc_cross_mode "none" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr5e-4_d2048_poly3_bs32)"

python ngrc_rope_lm_cross_low_rank.py --local_batch_size 32 --learning_rate 1e-3 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3 --ngrc_cross_mode "none" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_1BT_random_lr1e-3_d2048_poly3_bs32)"

