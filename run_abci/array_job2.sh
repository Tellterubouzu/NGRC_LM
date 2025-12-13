#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=2:30:00
#PBS -P gcc50527
#PBS -j oe
#PBS -m abe
#PBS -M shimomura.teruki174@mail.kyutech.jp
#PBS -J 1-50

module purge
module load cuda/12.6
module load cudnn/9.5
source ~/miniconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0
export CC=gcc
export CXX=g++
cd ${PBS_O_WORKDIR}
conda activate esn
cd ../src/ESN

# --------------------------------------------
# sigma_rec を job array の ID に基づいて決定
# --------------------------------------------
if [ ${PBS_ARRAY_INDEX} -le 50 ]; then
    sigma_rec=1.0
fi

echo "Job ${PBS_ARRAY_INDEX} running with sigma_rec=${sigma_rec}"

python3 experiment_initialize.py --sigma_rec ${sigma_rec}
