#!/bin/bash

scripts="python ./render/render_sb3_ppo.py"

# # Read the list of robots from the file

# if [ "$#" -ne 5 ]; then
#     echo "Usage: $0 <robots_json> <log_path> <env_id> <total_timesteps> <exp_name>"
#     exit 1
# fi

# # Assign arguments to variables
# robots_json="$1"
# log_path="$2"
# env_id="$3"
# total_timesteps="$4"

# # echo "This is $robots_list"
# # echo "Perf log path is $perf_log_path"

# output_dir="$log_path/output_files"
# mkdir -p "$output_dir"

# # Create a temporary SLURM script with the correct output path
# tmp_slurm_script=$(mktemp /tmp/render_sb3_sbatch_XXXXXX.sh)

# # Parse the JSON file, sort entries by robot_performance, and get the top 10 robots
# # top_robots=$(jq -r 'map(to_entries | .[0]) | sort_by(.value) | .[:10] | .[].key' "$robots_json")

# top_robots=("/home/knagiredla/gfn_archive/gfn_current/gfn_fixed_comp/logs/exp_betadecay_fxd_comp_2_1718065373/train/robot_4_1718262423.xml" "/home/knagiredla/gfn_archive/gfn_current/gfn_fixed_comp/logs/exp_betadecay_fxd_comp_2_1718065373/train/robot_4_1718262423.xml")
# # # Use Python to parse the JSON, sort by performance, and get the top 10 robots
# # top_robots=$(python3 - <<EOF
# # import json
# # import sys

# # # Read the input JSON file
# # with open("$robots_json") as f:
# #     data = json.load(f)

# # # print("data", data[0])

# # # Extract and sort robots by performance
# # sorted_robots = sorted(data, key=lambda x: list(x.values())[0])

# # # Get the top 10 robots
# # top_robots = sorted_robots[:10]

# # # Print the top 10 robot file paths
# # for robot in top_robots:
# #     print("PRINTING THIS LINE", list(robot.keys())[0])
# # EOF
# # )
# # echo "top robots are: $top_robots"

# # Loop over the top 10 robots and run the script 10 times for each robot
# echo "these are $top_robots"
# for robot in $top_robots; do
#     output_file="$output_dir/$(basename "$robot").out"
#     sed "s|__OUTPUT_PATH__|$output_file|g" ./scripts/render_sb3_sbatch.sh > "$tmp_slurm_script"
#     # Submit the job
#     # echo "robot: $robot"
#     job_id=$(sbatch "$tmp_slurm_script" "$scripts" --xml_file_path "$robot" --log_path "$log_path" --env_id "$env_id" --total_timesteps "$total_timesteps" | awk '{print $4}')
#     job_ids+=($job_id)
# done

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

# # Clean up temporary script
# rm -f "$tmp_slurm_script"
# rm -rf "$output_dir"


# Read the list of robots from the file

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <robot_list> <log_path> <env_id> <total_timesteps> <exp_name>"
    exit 1
fi

# Assign arguments to variables
robots_list="$1"
log_path="$2"
env_id="$3"
total_timesteps="$4"
exp_name="$5"

echo "Received robots_list ::: $robots_list"

# echo "This is $robots_list"
# echo "Perf log path is $perf_log_path"

output_dir="$log_path/output_files"
mkdir -p "$output_dir"

# Create a temporary SLURM script with the correct output path
tmp_slurm_script=$(mktemp /tmp/render_sb3_sbatch_XXXXXX.sh)

while IFS= read -r robot; do
    output_file="$output_dir/$(basename "$robot").out"
    sed "s|__OUTPUT_PATH__|$output_file|g" ./scripts/render_sb3_sbatch.sh > "$tmp_slurm_script"
    echo "THIS IS THE SLURM ROBOT $robot"
    # Submit the job
    job_id=$(sbatch "$tmp_slurm_script" "$scripts" --xml_file_path "$robot" --log_path "$log_path" --env_id "$env_id" --total_timesteps "$total_timesteps" --name "$exp_name" | awk '{print $4}')
    echo "Slurm: $job_id"
    job_ids+=($job_id)

done < "$robots_list"

# Wait for all submitted SLURM jobs to complete
while true; do
    all_done=true
    for job_id in "${job_ids[@]}"; do
        if squeue | grep -q "$job_id"; then
            all_done=false
            break
        fi
    done
    if [ "$all_done" = true ]; then
        break
    else
        sleep 10
    fi
done

# Clean up temporary script
rm -f "$tmp_slurm_script"

echo "SUCCESS ::: COMPLTED"

# rm -rf "$output_dir"
