import os
import argparse
import json

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import fcntl
import time
import numpy as np
from matplotlib import pyplot as plt


def main():
    # print("Entered runppo")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env_id', help='environment ID', default="Ant-v5")
    parser.add_argument('--total_timesteps', help='maximum step size', type=int, default=20000)
    # parser.add_argument('--network', help='path for data')
    parser.add_argument('--xml_file_path', help='path for xml', default='/home/knagiredla/gfn_archive/gfn_current/gfn_fixed_comp/assets/test_base_ant_wall.xml')
    parser.add_argument('--perf_log_path', help='path for xml', default='./logs')
    # parser.add_argument('--ctrl_cost_weight', help='ctrl cost weight for gym env')

    args = parser.parse_args()

    # Logger configuration
    config_name = args.perf_log_path
    tmp_path = config_name 
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # vec_env = DummyVecEnv([lambda: gym.make(args.env_id, xml_file=args.xml_file_path, render_mode="rgb_array")])
    
    env_id = args.env_id
    total_timesteps = int(args.total_timesteps)

    xml_path = args.xml_file_path

    policy_kwargs = dict(
        net_arch=[128, 128, 128, 128]
    )

    env = gym.make(env_id, xml_file=xml_path)
    # Instantiate the model
    model = PPO("MlpPolicy", env, verbose=1, batch_size=2048, learning_rate=0.0001, 
                clip_range=0.1, ent_coef=0.01, policy_kwargs=policy_kwargs)    
    model.set_logger(new_logger)

    # Train the model
    # try:
    _, full_ep_rew_list = model.learn(total_timesteps=total_timesteps)
    last_rew = full_ep_rew_list[-1]

    # print("full_ep_rew_list", full_ep_rew_list, len(full_ep_rew_list))
 
    count = len(full_ep_rew_list)

    ep_rew_list = full_ep_rew_list[-(count//4):]
 
 
    #Calculate gradient
    tim = np.arange(0,len(ep_rew_list))
    # get linear trend lines
    slope, intercept = np.polyfit(tim, ep_rew_list, 1)
    # print("slope", slope)
    if slope < 0 and last_rew < 0:
        mean_agg_reward = 0.00015
    else:
        mean_agg_reward = slope * 100 + last_rew

    fig=plt.figure(figsize=(12,6))

    ax1=plt.subplot(121)
    ax1.plot(tim,ep_rew_list,c='C0',lw=2,label='ep_rews')
    ax1.plot(tim,slope*tim+intercept,ls='dotted',c='C0',lw=2,label='Gradient=%1.2f'%slope)
    ax1.legend()
    graphs_save_path = os.path.join(config_name, 'graphs')
    if not os.path.exists(graphs_save_path):
        os.mkdir(graphs_save_path)
    g_postfix = int(time.time())
    plt.savefig(graphs_save_path + f"/graph_{slope}_{mean_agg_reward}_{g_postfix}.png", bbox_inches='tight')

        # Evaluate the policy
        # mean_reward, std_reward, episode_rewards = evaluate_policy(model, env, n_eval_episodes=10)
        # print("episode rewards", episode_rewards, type(episode_rewards))
    # except Exception as e:
    #     mean_agg_reward = 0.0001 
    
    errBool = False
    if mean_agg_reward == None:
        mean_agg_reward = 0.00015
        errBool = True
    with open(os.path.join(config_name, 'ppo_err.txt'), 'a') as f_err:
        # Acquire an exclusive lock on the file
        fcntl.flock(f_err, fcntl.LOCK_EX)
        try:
            # Write the dictionary entry to the file as a JSON string
            if errBool:
                f_err.write(f"Another failed robot:  + {xml_path}")
            entry = {args.xml_file_path: mean_agg_reward}
            f_err.write(f"{entry} + type: {type(mean_agg_reward)}\n")

        finally:
            # Release the lock
            fcntl.flock(f_err, fcntl.LOCK_UN)

    # Save the model if the mean reward is greater than 200
    postfix = int(time.time())
    # if mean_agg_reward > 100:
    #     model_save_path = os.path.join(config_name, 'good_models')
    #     # Check if the directory exists, and create it if it doesn't
    #     if not os.path.exists(model_save_path):
    #         os.mkdir(model_save_path)
    #     model.save(model_save_path + f"/model_{mean_agg_reward}_{postfix}")
        # if mean_reward > 7:
        #     video_folder = os.path.join(os.path.dirname(config_name), "videos")
        #     video_length = 100

        #     vec_env = DummyVecEnv([lambda: gym.make(env_id, xml_file=xml_path, render_mode="rgb_array")])

        #     # Record the video starting at the first step
        #     vec_env = VecVideoRecorder(vec_env, video_folder,
        #                         record_video_trigger=lambda x: x == 0, video_length=video_length,
        #                         name_prefix=f"agent-{mean_reward}-{postfix}")

        #     obs = vec_env.reset()

        #     for _ in range(video_length + 1):
        #         actions, _ = model.predict(obs)
        #         obs, rewards, dones, infos = vec_env.step(actions)
        #         vec_env.close()
                
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
    # print("mean_agg_reward", mean_agg_reward)

if __name__ == '__main__':
    main()
