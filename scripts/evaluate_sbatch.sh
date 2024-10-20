#!/bin/bash
#SBATCH --job-name=PPO_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --error=/home/knagiredla/gfn_archive/gfn_current/output_eval/error.err
#SBATCH --time=05:00:00
#SBATCH --output=/home/knagiredla/gfn_archive/gfn_current/output_eval/%j.out
#SBATCH --qos=batch-short

source ~/.bashrc
conda activate kishan
$@
