# (Optional) A name to describe the current running experiment
experiment_name="ant"

scripts="python ./tasks/tunePPO.py"  

for seed in 1
do
    for clip in 0.1 0.2 0.3 0.4
    do
        for ent_coef in 0.01
        do 
            for ((i = 0; i < ${#scripts[@]}; i++))
            do  
                sbatch ./scripts/PPO_sbatch.sh ${scripts[$i]} --seed=$seed --clip=$clip --ent_coef=$ent_coef
                echo ${scripts[$i]}
            done
        done
    done
done
