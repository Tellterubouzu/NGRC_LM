#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=4:00:00 
#PBS -W group_list=gp36
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp
#PBS -N ngrc_zz2_64

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


python ngrc_lm.py --local_batch_size 200 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 64 --ngrc_lag 16 --ngrc_feature z_z2

python ngrc_lm.py --local_batch_size 200 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 64 --ngrc_lag 32 --ngrc_feature z_z2

python ngrc_lm.py --local_batch_size 200 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 64 --ngrc_lag 64 --ngrc_feature z_z2

python ngrc_lm.py --local_batch_size 200 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 64 --ngrc_lag 128 --ngrc_feature z_z2

python ngrc_lm.py --local_batch_size 200 --learning_rate 5e-4 --seq_len 256 --ngrc_d_model 64 --ngrc_lag 256 --ngrc_feature z_z2
