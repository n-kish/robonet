import json
import argparse
import time
import multiprocessing
import subprocess
import os
import datetime
import numpy as np
import fcntl
import re

'''
This file:
1.) Takes as input a list of dictionaries (eg. rews.json) 
2.) Picks robots out and places in a seperate list - robot_list 
3.) Iterates over robot_list and places each robot in the simulator for 1M timesteps (with parallelization through python multiprocessing module)
4.) Collects each of these rewards and writes into a dictionary called 'result_perf' in the format {robot: reward}
5.) If possible, collect a screenshot of each of these robots, write reward value on it and save them in logdir. (TO DO)
'''

def modify_and_read_file(robots_path, robot):
    rews_file_path = os.path.join(robots_path, 'rews.json')
    modified_rews_file_path = os.path.join(robots_path, 'rew_modified.json')

    # Lock and read the original file
    with open(rews_file_path, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # Acquire a shared lock for reading
        content = f.read()
        fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock

    # Modify the content
    # Replace sequences of commas and newlines with a single comma
    content = re.sub(r'[,\n]+', ',', content)

    # Trim any leading or trailing commas
    content = content.strip(',')

    # Modify the content to add '[' at the start and ']' at the end
    modified_content = '[' + content + ']'

    # Lock and write the modified content to a new file
    with open(modified_rews_file_path, 'w') as f1:
        fcntl.flock(f1, fcntl.LOCK_EX)  # Acquire an exclusive lock for writing
        f1.write(modified_content)
        fcntl.flock(f1, fcntl.LOCK_UN)  # Release the lock

    # Lock and read the modified JSON file
    with open(modified_rews_file_path, 'r') as f2:
        fcntl.flock(f2, fcntl.LOCK_SH)  # Acquire a shared lock for reading
        modified_data = json.load(f2)
        fcntl.flock(f2, fcntl.LOCK_UN)  # Release the lock

    eprewmean = None
    for list_elem in modified_data:
        if robot in list_elem:
            eprewmean = list_elem[robot]
            break

    return eprewmean


def write_robots_to_file(robots, robots_path, filename):
    with open(os.path.join(f"{robots_path}", filename), "w") as file:
        for robot in robots:
            file.write(f"{robot}\n")

def call_train_script(robots_path):

    env_id = "Ant-v5"
    timesteps = str(10000000)
    ppo_script = "python ./tasks/ppoeval_sb3.py"

    args1= f"{robots_path}" + "gen_robots_list.txt" 
    args2= f"{robots_path}"
    args3= f"{env_id}"
    args4= f"{timesteps}"
    args5= f"{ppo_script}"

    print("args1", args1) 
    print("args2", args2)
    print("args3", args3)
    print("args4", args4)
    print("args5", args5)

    args6= 0.0005
    
    result = subprocess.run(['bash', './scripts/gfn_sb3_ppo.sh', args1, args2, args3, args4, args5, args6], check=True)
    
    if result.returncode == 0:
        pass
    else:
        print(f"train.sh execution failed with return code {result.returncode}")
        print(f"Standard Output: {result.stdout}")
        print(f"Standard Error: {result.stderr}")


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rews_path', help='Path of rews.json file', type=str, default=None)
    parser.add_argument('--folder_path', help='Path of folder containing all sampled robots', type=str, default=None)
    args = parser.parse_args()

    folder_path = args.folder_path  

    # Example folder_path: '/home/knagiredla/gflownet/src/gflownet/logs/exp_ant_bal_1_1709549428/xmlrobots/gen_5_steps/final'

    if args.rews_path and folder_path is None:
        print("Missing arguments, either specify rews.json path (for already trained robots) or folder_path (for fresh samples)")
        exit()
    
    print("**************************************************************")

    if folder_path is not None:
        print("Path", folder_path)

    # Collect Robots from given rews.json file path
    max_eprewmean = float('-inf')  # Initialize with a very small value
    max_dict = None

    robot_list = []
    data = []
    # print("args.rew_path", args.rews_path)

    #TO DO: Change the way rews.json is being written in run_ppo.py to make these following lines simpler.
    if args.rews_path is not None: 
        with open(args.rews_path, 'r') as file:
            for line in file:
                try:
                    all_robots_perf = json.loads(line)
                    for robot_perf in all_robots_perf:
                        if "xml_path" in robot_perf:
                            robot = robot_perf['xml_path']
                            # print("curr_robot", robot)
                            robot_list.append(robot)
                except json.JSONDecodeError as e:
                    print(f"HIT expection because of {e}")
    else:
        robots_path = folder_path+'/valid/'
        robot_list = [robots_path+robot for robot in os.listdir(robots_path) if robot.endswith('.xml')]
    print("robots path", robots_path)
    write_robots_to_file(robot_list, robots_path, filename="gen_robots_list.txt")
    call_train_script(robots_path)
    print("out of call_train_script")

    eprewmeans = []
    no_return_value = 0.001
    for robot in robot_list:
        eprewmean = modify_and_read_file(robots_path, robot)
        if eprewmean == None:
            print("None as eprewmean & robot", eprewmean, robot)
            eprewmeans.append(no_return_value)
        else:
            eprewmeans.append(eprewmean)

    print("eprewmeans", eprewmeans)

    eprewmeans.sort(reverse = True)

    result_perf = {}
    result_perf = {robot: eprewmeans for robot, eprewmeans in zip(robot_list, eprewmeans)}

    # print("result_perf", result_perf)

    with open(f'{robots_path}' + 'final_gen_result.json', 'w') as fp:
        json.dump(result_perf, fp)

    print("mean & std.dev of all robots", np.mean(eprewmeans), np.std(eprewmeans))

    print("mean & std.dev of top-10 robots", np.mean(eprewmeans[:10]), np.std(eprewmeans[:10]))

    print("Top Performer", max(result_perf, key=result_perf.get), max(result_perf.values()))

    print("**************************************************************")

    # TO DO: An idea is to collect the results from result_perf and save an image of the robot and impose the pred_rew on the image
    # (Currently implemented) To write result_perf to a file. 

if __name__ == "__main__":
    main()
