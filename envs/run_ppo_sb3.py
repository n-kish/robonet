import os
import argparse
import json

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv



def main():
    # print("Entered runppo")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env_id', help='environment ID', default=None)
    parser.add_argument('--total_timesteps', help='maximum step size', type=int)
    parser.add_argument('--network', help='path for data')
    parser.add_argument('--xml_file_path', help='path for xml')
    parser.add_argument('--perf_log_path', help='path for xml')
    parser.add_argument('--ctrl_cost_weight', help='ctrl cost weight for gym env')

    args = parser.parse_args()

    # Logger configuration
    config_name = args.perf_log_path
    tmp_path = config_name
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # vec_env = DummyVecEnv([lambda: gym.make(args.env_id, xml_file=args.xml_file_path, render_mode="rgb_array")])
     
    env = gym.make(args.env_id, xml_file=args.xml_file_path)
    # Instantiate the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.set_logger(new_logger)

    # Train the model
    model.learn(total_timesteps=int(args.total_timesteps))

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Save the model if the mean reward is greater than 200
    if mean_reward > 200:
        model.save(args.perf_log_path)

    data = {args.xml_file_path: mean_reward}
    with open(os.path.join(config_name, 'rews.json'), 'a') as outfile:
        json.dump(data, outfile)
        outfile.write(',')


if __name__ == '__main__':
    main()
