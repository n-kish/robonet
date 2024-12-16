#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=ppo_30min
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --error="__ERROR_PATH__/ppo_%A_%a.err"
#SBATCH --time=02:00:00
#SBATCH --output="__OUTPUT_PATH__/ppo_%A_%a.out"
#SBATCH --prefer=cpu-amd

source ~/.bashrc
conda activate gfn_sb3_gpu

# Read the specific line from the robots list file based on array task ID
ROBOT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")

# Execute the command with the robot from the array
$2 --xml_file_path "$ROBOT" --perf_log_path "$3" --env_id "$4" --min_timesteps "$5" --ctrl_cost_weight "$6"