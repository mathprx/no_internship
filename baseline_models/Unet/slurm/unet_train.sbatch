#!/bin/bash

#SBATCH --mail-type ALL
#SBATCH --mail-user mathieu.prioux@unil.ch
#SBATCH --account tbeucler_ai4pex

#SBATCH --partition gpu

#SBATCH -t 30:00
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 24
#SBATCH --gpus-per-node 2

#SBATCH --job-name unet_train
#SBATCH --output unet_train_out.out

##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_unet.py --config-name=config"

module load miniforge3
conda_init
conda activate venv2
module load cuda

source ddp_export.sh

cd ..
srun -n $SLURM_NTASKS python train_unet.py --config-name=config
