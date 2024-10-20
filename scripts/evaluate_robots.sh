lower_limit=400
upper_limit=400
step=200
scripts="python tasks/evaluate_robots.py"

for (( it = $lower_limit; it <= $upper_limit; it += $step ))
do
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch ./scripts/evaluate_sbatch.sh ${scripts[$i]} --folder_path "/home/knagiredla/gfn_archive/gfn_current/gfn_fixed_comp/logs/exp_orig1_75_300k_gsca_flat_3_1725246076/xmlrobots/gen_${it}_steps"
    done
done
