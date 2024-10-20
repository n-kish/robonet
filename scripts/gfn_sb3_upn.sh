#!/bin/bash

# Read the list of robots from the file

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <robot_list> <encodings_list> <perf_log_path> <env_id> <total_timesteps> <scripts>"
    exit 1
fi

# Assign arguments to variables
robots_list="$1"
encodings_list="$2"
perf_log_path="$3"
env_id="$4"
total_timesteps="$5"
scripts="$6"

# echo "This is $robots_list"
# echo "Perf log path is $perf_log_path"

output_dir="$perf_log_path/output_files"
mkdir -p "$output_dir"

# Create a temporary SLURM script with the correct output path
tmp_slurm_script=$(mktemp /tmp/gfn_sb3_sbatch_XXXXXX.sh)
# echo "tmp_slurm_script is $tmp_slurm_script"

# Read robots and encodings into arrays
readarray -t robots < "$robots_list"
IFS=',' read -ra encodings <<< "$encodings_list"

for robot in "${robots[@]}"; do
    for encoding in "${encodings[@]}"; do
        output_file="$output_dir/$(basename "$robot")_${encoding}.out"
        sed "s|__OUTPUT_PATH__|$output_file|g" ./scripts/gfn_sb3_sbatch.sh > "$tmp_slurm_script"
        
        # Submit the job
        job_id=$(sbatch "$tmp_slurm_script" "$scripts" --xml_file_path "$robot" --perf_log_path "$perf_log_path" --env_id "$env_id" --total_timesteps "$total_timesteps" --encoding "$encoding" | awk '{print $4}')
        # echo "Job id is $job_id"
        job_ids+=($job_id)
    done
done

# # Wait for all submitted SLURM jobs to complete
# while true; do
#     all_done=true
#     for job_id in "${job_ids[@]}"; do
#         if squeue | grep -q "$job_id"; then
#             all_done=false
#             break
#         fi
#     done
#     if [ "$all_done" = true ]; then
#         break
#     else
#         sleep 10
#     fi
# done


#!/bin/bash

# Set the timeout in seconds
job_timeout=$((60 * 60))         # 30 minutes for running jobs
queue_timeout=$((60 * 60))       # 1 hour for pending jobs

# Declare an associative array to track job start times (both pending and running)
declare -A job_start_times

# Wait for all submitted SLURM jobs to complete, with a 30-minute timeout per job
while true; do
    all_done=true
    current_time=$(date +%s)

    for job_id in "${job_ids[@]}"; do
        # Get the job status (e.g., RUNNING, PENDING, COMPLETED)
        job_status=$(squeue -j "$job_id" -h -o "%T")

        if [ "$job_status" = "RUNNING" ]; then
            all_done=false

            # If the job hasn't been tracked yet, record its start time
            if [ -z "${job_start_times[$job_id]}" ]; then
                job_start_times[$job_id]=$current_time
                # echo "Tracking start time for running job $job_id"
            fi

            # Calculate how long the job has been running
            job_start_time=${job_start_times[$job_id]}
            elapsed_time=$((current_time - job_start_time))

            # If the job has been running for more than the timeout, cancel it
            if [ "$elapsed_time" -ge "$job_timeout" ]; then
                echo "Job $job_id has exceeded the 30-minute running timeout. Cancelling."
                scancel "$job_id"
            fi

        elif [ "$job_status" = "PENDING" ]; then
            all_done=false

            # If the job hasn't been tracked yet, record its queue entry time
            if [ -z "${job_start_times[$job_id]}" ]; then
                job_start_times[$job_id]=$current_time
                # echo "Tracking start time for pending job $job_id"
            fi

            # Calculate how long the job has been waiting in the queue
            job_start_time=${job_start_times[$job_id]}
            elapsed_time=$((current_time - job_start_time))

            # If the job has been waiting in the queue for more than 1 hour, cancel it
            if [ "$elapsed_time" -ge "$queue_timeout" ]; then
                echo "Job $job_id has been in the queue for over an hour. Cancelling."
                scancel "$job_id"
            fi
        fi
    done

    # If all jobs are done or cancelled, exit the loop
    if [ "$all_done" = true ]; then
        # echo "All jobs completed or cancelled."
        break
    else
        sleep 10  # Sleep for 10 seconds before checking again
    fi
done


# Clean up temporary script
rm -f "$tmp_slurm_script"
rm -rf "$output_dir"


# echo "All ppo_sb3.py processes have completed"
