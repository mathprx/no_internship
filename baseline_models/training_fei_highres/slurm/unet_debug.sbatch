#!/bin/bash

#SBATCH --mail-type ALL
#SBATCH --mail-user fangfei.lan@unil.ch
#SBATCH --account tbeucler_ai4pex

#SBATCH --partition gpu

#SBATCH -t 30:00
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 24
#SBATCH --gpus-per-node 2

#SBATCH --job-name unet_debug
#SBATCH --output unet_debug_out.out

##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_unet.py --config-name=config_debug"

module load miniforge3
conda_init
conda activate hybridesmbench
module load cuda

source ddp_export.sh

cd ..
srun -n $SLURM_NTASKS python train_unet.py --config-name=config_debug
