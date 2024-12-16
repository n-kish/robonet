import os
import argparse
import json

import gymnasium as gym
from sbx import PPO
# from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv  # Add this import
import multiprocessing
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean



import fcntl
import time
import re
import math
import numpy as np
from matplotlib import pyplot as plt

def main():
    # print("Entered runppo")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env_id', help='environment ID', default=None)
    parser.add_argument('--min_timesteps', help='maximum step size', type=int)
    # parser.add_argument('--network', help='path for data')
    parser.add_argument('--xml_file_path', help='path for xml')
    parser.add_argument('--perf_log_path', help='path for xml')
    parser.add_argument('--ctrl_cost_weight', help='ctrl cost weight for gym env', type=float, default=0.0005)

    args = parser.parse_args()

    # Logger configuration
    config_name = args.perf_log_path
    tmp_path = config_name 
    # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # vec_env = DummyVecEnv([lambda: gym.make(args.env_id, xml_file=args.xml_file_path, render_mode="rgb_array")])
    
    env_id = args.env_id
    min_steps = int(args.min_timesteps)
    robot = args.xml_file_path

    # Resource Allocation strategy
    # pattern = r"robot_(\d+)_"
    pattern = re.compile(r'robot_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.xml')

    # Use re.search to find the pattern in the string
    match = re.search(pattern, robot)
    # Extract the information using group() method
    node_count = int(match.group(1))  # Extracts the node count
    time_steps = int(float(match.group(2)))
    
    # time_scaler = node_count * 0.1
    # cost_scaler = 10* math.log((time_steps+1))
    cost_scalar = abs(0.002 * (time_steps - min_steps * node_count))      # calculates base timesteps and sampled timesteps difference
    
    # calc_timesteps = min_steps + (time_scaler * min_steps)            #Resource allocation

    policy_kwargs = dict(
        net_arch=[128, 128, 128, 128]
    )
    # print("time_steps", time_steps)
    path = args.perf_log_path
    n_envs = 4
    vec_env = SubprocVecEnv([
        lambda i=i: Monitor(
            gym.make(env_id,
                xml_file=robot,
                render_mode='rgb_array'),
            os.path.join(path, str(i))
        ) for i in range(n_envs)
    ])

    # env = gym.make(env_id, xml_file=robot)
    # Instantiate the model
    model = PPO("MlpPolicy", vec_env, verbose=1, batch_size=2048, learning_rate=0.0001, 
                clip_range=0.1, ent_coef=0.01, policy_kwargs=policy_kwargs)    
    # model.set_logger(new_logger)

    full_ep_rew_list = []
    # Add callback to collect rewards during training
    class RewardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.ep_rewards_history = []

        def _on_step(self) -> bool:
            # Required method that gets called at every step
            return True

        def _on_rollout_end(self):
            # Collect rewards at the end of each rollout (episode)
            if len(self.model.ep_info_buffer) > 0:
                self.ep_rewards_history.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
            return True 

    reward_callback = RewardCallback()
    
    # Pass callback to learn
    model.learn(total_timesteps=time_steps, progress_bar=True, callback=reward_callback)    # mean_ep_reward = np.mean(ep_rew_list)
    
    full_ep_rew_list = reward_callback.ep_rewards_history

    last_rew = full_ep_rew_list[-1]

    entry0 = {args.xml_file_path: last_rew}
    # Open the file in append mode
    with open(os.path.join(config_name, 'pre_rews.json'), 'a') as f:
        # Acquire an exclusive lock on the file
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Write the dictionary entry to the file as a JSON string
            f.write(json.dumps(entry0) + "\n" + ",")
        finally:
            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)   
    # print("last_rew", last_rew)
    #Calculate gradient
    # tim = np.arange(0,len(ep_rew_list[-3:]))
    # # get linear trend lines
    # slope, intercept = np.polyfit(tim, ep_rew_list[-3:], 1)

    if last_rew < 0:
        mean_agg_reward = 0.00015
    else:
        mean_agg_reward = last_rew + 200
        
    # Evaluate the policy
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    mean_agg_reward = mean_agg_reward - cost_scalar              #Cost allocation
    # except Exception as e:
    #     mean_agg_reward = 0.0001 
    errBool = False
    if mean_agg_reward == None:
        mean_agg_reward = 0.00015
        errBool = True
    # elif mean_agg_reward < 0:
    #     mean_agg_reward = 0.00019
    with open(os.path.join(config_name, 'ppo_err.txt'), 'a') as f_err:
        # Acquire an exclusive lock on the file
        fcntl.flock(f_err, fcntl.LOCK_EX)
        try:
            # Write the dictionary entry to the file as a JSON string
            if errBool:
                f_err.write(f"Another failed robot:  + {robot}")
            entry = {args.xml_file_path: mean_agg_reward}
            f_err.write(f"{entry} + type: {type(mean_agg_reward)}\n")
        finally:
            # Release the lock
            fcntl.flock(f_err, fcntl.LOCK_UN)

    # Save the model if the mean reward is greater than 200
    # postfix = int(time.time())
    # if mean_agg_reward > 500:
    #     model_save_path = os.path.join(config_name, 'good_models')
    #     # Check if the directory exists, and create it if it doesn't
    #     if not os.path.exists(model_save_path):
    #         os.mkdir(model_save_path)
    #     model.save(model_save_path + f"/model_{mean_agg_reward}_{postfix}")
                
    # Log the results
    entry = {args.xml_file_path: mean_agg_reward}
    # Open the file in append mode
    with open(os.path.join(config_name, 'rews.json'), 'a') as f:
        # Acquire an exclusive lock on the file
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Write the dictionary entry to the file as a JSON string
            f.write(json.dumps(entry) + "\n" + ",")
        finally:
            # Release the lock
            fcntl.flock(f, fcntl.LOCK_UN)

if __name__ == '__main__':
    main()
