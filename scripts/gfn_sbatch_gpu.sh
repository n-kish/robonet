#!/bin/bash
#SBATCH --qos=batch-long
#SBATCH --job-name=gfn_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --error="/scratch/knagiredla/robonet/output/gfn_simtest.err"
#SBATCH --time=200:00:00
#SBATCH --output=/scratch/knagiredla/robonet/output/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=knagiredla@deakin.edu.au



source ~/.bashrc
conda activate gfn_sb3_gpu
$@
