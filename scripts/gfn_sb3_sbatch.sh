#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=gfn_30min
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --error="/home/knagiredla/gfn_archive/gfn_current/output/ppo.err"
#SBATCH --time=02:00:00
#SBATCH --output=__OUTPUT_PATH__


source ~/.bashrc
conda activate gfn_sb3
$@
