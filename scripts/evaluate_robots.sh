lower_limit=500
upper_limit=500
step=200
scripts="python tasks/evaluate_robots.py"

for (( it = $lower_limit; it <= $upper_limit; it += $step ))
do
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch ./scripts/evaluate_sbatch.sh ${scripts[$i]} --folder_path "/home/knagiredla/robonet/logs/exp_GSCA_10_flat_base_ant_40_000_134_1729566757/xmlrobots/gen_${it}_steps"
    done
done
