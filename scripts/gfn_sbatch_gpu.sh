#!/bin/bash
#SBATCH --qos=batch-long
#SBATCH --job-name=gfn_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=5G
#SBATCH --error="/home/knagiredla/robonet/output/gfn_simtest.err"
#SBATCH --time=240:00:00
#SBATCH --output=/home/knagiredla/robonet/output/%j.out



source ~/.bashrc
conda activate gfn_sb3
$@
