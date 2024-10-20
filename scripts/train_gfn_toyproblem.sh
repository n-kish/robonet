experiment_name="Toy_longbots"

scripts="python ./tasks/train_robot_height.py"  

for seed in 2
do
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch ./scripts/gfn_sbatch.sh ${scripts[$i]} --name $experiment_name
        echo ${scripts[$i]} --name $experiment_name
    done
done
