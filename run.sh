#!/bin/bash
#SBATCH --job-name=HeckYes
#SBATCH -t 18:06:00
#SBATCH -N 1
#SBATCH --partition=gpu_shared

module load 2020
module load eb
module load Python/3.8.2-GCCcore-9.3.0 
module load cuDNN/8.0.3.33-gcccuda-2020a 
module load NCCL/2.7.8-gcccuda-2020a 


