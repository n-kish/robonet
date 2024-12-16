
seed=134
env="ant"
env_id="Ant-v5"
train_steps=500_000
start_point="base"    # "orig" / "base"
exp_method="GSCA"      # "CA" / "GSCA" / "linearscaling"
env_terrain="incline"      # "flat" / "incline" / "gap" / "wall" / "rugged"
path="/scratch/knagiredla/robonet/logs"
max_nodes=10
min_steps=50_000
global_batch_size=64

# echo "env_terrain is $env_terrain"

if [[ "$env_terrain" == "gap" || "$env_terrain" == "wall" || "$env_terrain" == "rugged" ]]; then
    ext_terrain=1   #True
else
    ext_terrain=0   #False
fi 

# echo "ext_terrain is $ext_terrain"

experiment_name="${exp_method}_${max_nodes}_${env_terrain}_${start_point}_${env}_${min_steps}"

scripts="python ./tasks/train_gfn.py"  

for seed in 134
do
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch ./scripts/gfn_sbatch_gpu.sh ${scripts[$i]} --name $experiment_name \
            --rl_timesteps $train_steps \
            --min_steps $min_steps \
            --lastbatch_rl_timesteps 1_000_000 \
            --seed $seed \
            --env_id $env_id \
            --env $env \
            --start_point $start_point \
            --env_terrain $env_terrain \
            --exp_method $exp_method \
            --base_xml_path "./assets/${start_point}_${env}_${env_terrain}.xml" \
            --terrain_from_external_source $ext_terrain \
            --run_path $path \
            --max_gfn_nodes $max_nodes \
            --seed $seed \
            --global_batch_size $global_batch_size
        echo ${scripts[$i]} --name $experiment_name --rl_timesteps $rl_timesteps --terrain_from_external_source $ext_terrain
    done
done
