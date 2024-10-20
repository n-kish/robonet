# (Optional) A name to describe the current running experiment
experiment_name="Incline_gsca_baseant"

scripts="python ./tasks/train_costaware_gfn.py"  

for seed in 3
do
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch ./scripts/gfn_sbatch.sh ${scripts[$i]} --name $experiment_name --seed=$seed --base_xml_path './assets/base_incline_ant.xml' --wall_terrain False
        echo ${scripts[$i]} --name $experiment_name
    done
done
