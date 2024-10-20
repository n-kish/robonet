
seed=0
env="ant"
train_steps=5_000
start_point="base"    # "orig" / "base"
exp_method="UPN"      # "naive" / "GS" / "CA" / "GSCA"   
env_terrain="incline"      # "flat" / "incline" / "gap" / "wall"
path="/home/knagiredla/gfn_archive/gfn_current/expt_logs"
max_nodes=3
min_steps=5_000

echo "env_terrain is $env_terrain"

if [[ "$env_terrain" == "gap" || "$env_terrain" == "wall" ]]; then
    ext_terrain=1   #True
else
    ext_terrain=0   #False
fi 

echo "ext_terrain is $ext_terrain"

experiment_name="${exp_method}_${max_nodes}_${env_terrain}_${start_point}_${env}_${train_steps}"

scripts="python ./tasks/train_gfn_upn.py"  

for seed in 2
do
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch ./scripts/gfn_sbatch.sh ${scripts[$i]} --name $experiment_name \
            --rl_timesteps $train_steps \
            --min_steps $min_steps \
            --lastbatch_rl_timesteps 1_000_000 \
            --seed $seed \
            --env $env \
            --start_point $start_point \
            --env_terrain $env_terrain \
            --exp_method $exp_method \
            --base_xml_path "./assets/${start_point}_${env}_${env_terrain}.xml" \
            --terrain_from_external_source $ext_terrain \
            --run_path $path \
            --max_gfn_nodes $max_nodes \
            --seed $seed
        echo ${scripts[$i]} --name $experiment_name --rl_timesteps $rl_timesteps --terrain_from_external_source $ext_terrain
    done
done
