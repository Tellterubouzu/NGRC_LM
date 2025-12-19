#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=24:00:00 
#PBS -W group_list=gp36
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp
#PBS -N ngrc_1_64dim

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
# 113.51M_d2048_lag10_poly3_rank512_lr0.0005_bs150_seq256_20251217-142020

python ngrc_lm_low_rank.py --local_batch_size 352 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 512 --ngrc_lag 16 --ngrc_poly_degree 3 --log_grad --total_tokens 5e9


#python ngrc_lm_low_rank.py --local_batch_size 160 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3

#python ngrc_lm_low_rank.py --local_batch_size 192 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3

#python ngrc_lm_low_rank.py --local_batch_size 224 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3

#python ngrc_lm_low_rank.py --local_batch_size 256 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 2048 --ngrc_lag 10 --ngrc_poly_degree 3

