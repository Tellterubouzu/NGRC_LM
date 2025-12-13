#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=00:45:00
#PBS -P gcc50527
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp

module purge
module load cuda/12.6
module load cudnn/9.5
source ~/miniconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0
export CC=gcc
export CXX=g++
cd ${PBS_O_WORKDIR}
conda activate esn
cd ../src/Multi_readout_ESN

nvidia-smi

pip install matplotlib seaborn pandas
#python3 experiment_initialize.py --sigma_rec ${sigma_rec}

#python3 make_csv.py 
#python3 metrics.py
#python3 analyze.py
python3 plot_bp.py
