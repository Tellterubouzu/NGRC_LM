
#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=12:00:00 
#PBS -W group_list=gp36
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp
#PBS -N gpt_10BT

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
python gpt_128M.py --n_embd 2048 --n_layer 1 --n_head 8 --local_batch_size 200 --learning_rate 5e-4 --seq_len 512 --wandb_run_name "SingleGPT(RoPE)115.9M_10BToken_size110_seq_len512)1"  --total_tokens 10e9
python gpt_128M.py --n_embd 2048 --n_layer 1 --n_head 8 --local_batch_size 150 --learning_rate 5e-4 --seq_len 512 --wandb_run_name "SingleGPT(RoPE)115.9M_10BToken_size110_seq_len512)1"  --total_tokens 10e9
python gpt_128M.py --n_embd 2048 --n_layer 1 --n_head 8 --local_batch_size 110 --learning_rate 5e-4 --seq_len 512 --wandb_run_name "SingleGPT(RoPE)115.9M_10BToken_size110_seq_len512)1"  --total_tokens 10e9
