#!/bin/bash
#SBATCH --job-name=gfn_render
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --error="/home/knagiredla/gfn_archive/gfn_current/output/ppo_render.err"
#SBATCH --time=05:00:00
#SBATCH --output=__OUTPUT_PATH__
#SBATCH --qos=batch-short

source ~/.bashrc
conda activate gfn_sb3
$@
