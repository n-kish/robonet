#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=gfn_30min
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --error="/home/knagiredla/robonet/output/ppo.err"
#SBATCH --time=02:00:00
#SBATCH --output=__OUTPUT_PATH__
#SBATCH --prefer=cpu-amd  # Request the cpu-amd feature
#SBATCH --nodelist=cpu-f-04



source ~/.bashrc
conda activate gfn_sb3_gpu
$@
