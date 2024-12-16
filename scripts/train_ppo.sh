#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <robot_list> <perf_log_path> <env_id> <min_timesteps> <scripts> <ctrl_cost_weight>"
    exit 1
fi

# Assign arguments to variables
robots_list="$1"
perf_log_path="$2"
env_id="$3"
min_timesteps="$4"
scripts="$5"
ctrl_cost_weight="$6"

# Create output directories
output_dir="$perf_log_path/output_files"
error_dir="$perf_log_path/error_files"
mkdir -p "$output_dir" "$error_dir"

# Count number of robots (lines) in the input file
n_tasks=$(wc -l < "$robots_list")

# Create a temporary SLURM script with the correct output path
tmp_slurm_script=$(mktemp /tmp/train_sbatch_array_XXXXXX.sh)

# Replace placeholders in the template
sed -e "s|__OUTPUT_PATH__|$output_dir|g" \
    -e "s|__ERROR_PATH__|$error_dir|g" \
    -e "/#SBATCH --prefer=cpu-amd/a#SBATCH --array=0-$((n_tasks-1))%32" \
    ./scripts/train_sbatch_array.sh > "$tmp_slurm_script"

# Submit the array job
job_id=$(sbatch "$tmp_slurm_script" "$robots_list" "$scripts" "$perf_log_path" "$env_id" "$min_timesteps" "$ctrl_cost_weight" | awk '{print $4}')

# Monitor the array job with timeout
job_timeout=$((60 * 60))    # 1 hour for running jobs
queue_timeout=$((60 * 60))  # 1 hour for pending jobs
start_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    
    # Check job status
    job_status=$(squeue -j "$job_id" -h -o "%T")
    
    # If no status, job is complete
    if [ -z "$job_status" ]; then
        break
    fi
    
    # Check timeouts
    elapsed_time=$((current_time - start_time))
    
    if [ "$job_status" = "RUNNING" ] && [ "$elapsed_time" -ge "$job_timeout" ]; then
        echo "Array job $job_id has exceeded the running timeout. Cancelling."
        scancel "$job_id"
        break
    elif [ "$job_status" = "PENDING" ] && [ "$elapsed_time" -ge "$queue_timeout" ]; then
        echo "Array job $job_id has been in queue too long. Cancelling."
        scancel "$job_id"
        break
    fi
    
    sleep 10
done

# Clean up
rm -f "$tmp_slurm_script"
rm -rf "$output_dir" "$error_dir"