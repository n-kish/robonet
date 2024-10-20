#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --error="/home/knagiredla/gflownet/src/gflownet/output/PPO.err"
#SBATCH --time=01:30:00

#SBATCH --output=/home/knagiredla/gflownet/src/gflownet/output/%j.out

source ~/.bashrc
conda activate kishan
$@
