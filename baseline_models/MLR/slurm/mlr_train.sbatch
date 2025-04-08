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

#SBATCH --job-name mlr_train
#SBATCH --output mlr_train_out.out

##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err


module load miniforge3
conda_init
conda activate no_training
module load cuda/12.6.2

source ../ddp_export.sh

cd ..
srun -n $SLURM_NTASKS python train_mlr.py --config-name=config
