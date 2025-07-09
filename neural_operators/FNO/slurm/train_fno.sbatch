#!/bin/bash -l

#SBATCH --mail-type ALL
#SBATCH --mail-user mathieu.prioux@unil.ch
#SBATCH --account tbeucler_ai4pex

#SBATCH --partition gpu

#SBATCH -t 24:00:00
#SBATCH --mem 128G
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1

#SBATCH --job-name train_fno
#SBATCH --output train_fno_out.out
##SBATCH --error=train_fno_err.err

module purge
module load miniforge3
conda_init
module load cuda/12.6.2
source ../ddp_export.sh

conda activate no_training
echo $CONDA_DEFAULT_ENV
cd ..
python train_fno.py 