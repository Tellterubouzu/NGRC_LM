
#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=24:00:00 
#PBS -W group_list=gp36
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp
#PBS -N ngrc_non1_10B

#set -x  # 実行トレース
#echo "Shell flags: $-"
module purge
module load cuda/12.8
module load cudnn/9.10.1.4

export CC=gcc
export CXX=g++
export CUDA_VISIBLE_DEVICES=0

cd ${PBS_O_WORKDIR}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../../src
python ngrc_rope_lm_cross_low_rank.py --local_batch_size 192 --learning_rate 5e-4 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 1 --ngrc_cross_mode "none" --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_10BT_nonecross_lr5e-4_d2048_poly1_bs198)"

python ngrc_rope_lm_cross_low_rank.py --local_batch_size 192 --learning_rate 1e-3 --seq_len 512 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 1 --ngrc_cross_mode "none"  --total_tokens 10e9 --wandb_run_name "NGRC_LM_ROPE(113.51M_10BT_nonecross_lr1e-3_d2048_poly1_bs198)"
