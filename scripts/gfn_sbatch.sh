#!/bin/bash
#SBATCH --qos=batch-long
#SBATCH --job-name=gfn_full
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --error="/scratch/knagiredla/robonet/output/gfn_simtest.err"
#SBATCH --time=150:00:00
#SBATCH --output=/scratch/knagiredla/robonet/output/%j.out



source ~/.bashrc
conda activate gfn_sb3_gpu
$@
