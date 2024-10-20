import ast
import copy
import json
import os
import pathlib
import shutil
import socket
from typing import Any, Callable, Dict, List, Tuple, Union
import multiprocessing
import time
import argparse
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch_geometric.data as gd
# from rdkit import RDLogger
# from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset
import wandb
wandb.login()
import re
import math
import subprocess
import xml.etree.ElementTree as ET

from utils.misc import create_logger
from algo.flow_matching import FlowMatching
from algo.trajectory_balance import TrajectoryBalance
from data.replay_buffer import ReplayBuffer
from envs.frag_mol_env import FragMolBuildingEnvContext
from envs.graph_building_env import GraphBuildingEnv
from models import bengio2021flow
from models.graph_transformer import GraphTransformerGFN
from train import FlatRewards, GFNTask, GFNTrainer, RewardScalar
from utils.transforms import thermometer
from train import cycle


from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import fcntl


POLICY_PATH = ""
EXP_METHOD = ""


def modify_and_read_file(perf_path, robot_path):
    rews_file_path = os.path.join(perf_path, 'rews.json')
    modified_rews_file_path = os.path.join(perf_path, 'rew_modified.json')

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
        if robot_path in list_elem:
            eprewmean = list_elem[robot_path]
            break

    return eprewmean

def write_robots_to_file(robots, filename="robots_list.txt"):
    with open(os.path.join(f"{POLICY_PATH}", filename), "w") as file:
        for robot in robots:
            file.write(f"{robot}\n")


def call_train_script(perf_log_path, timesteps, env_id):

    env_id = "Ant-v5"
    timesteps = str(timesteps)
    # print("timesteps", timesteps, type(timesteps))
    ppo_script = f"python ./tasks/ppo_sb3_{EXP_METHOD}.py"

    args1= f"{POLICY_PATH}/robots_list.txt"
    args2= f"{perf_log_path}"
    args3= f"{env_id}"
    args4= f"{timesteps}"
    args5= f"{ppo_script}"
    
    result = subprocess.run(['bash', './scripts/gfn_sb3_ppo.sh', args1, args2, args3, args4, args5], check=True)
    
    if result.returncode == 0:
        pass
    else:
        print(f"train.sh execution failed with return code {result.returncode}")
        print(f"Standard Output: {result.stdout}")
        print(f"Standard Error: {result.stderr}")


class RoboGenTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching (TODO: port to this repo).
    """

    def __init__(
        self,
        log_dir,
        dataset: Dataset,
        temperature_distribution: str,
        temperature_parameters: Tuple[float, float],
        num_thermometer_dim: int,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        # self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim
        self.log_dir = log_dir

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        beta = None
        if self.temperature_sample_dist == "constant":
            assert type(self.temperature_dist_params) is float
            beta = np.array(self.temperature_dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            if self.temperature_sample_dist == "gamma":
                loc, scale = self.temperature_dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.temperature_sample_dist == "uniform":
                beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = self.temperature_dist_params[1]
                # print("BETA", beta)
                # print("upper_bound", upper_bound)
            elif self.temperature_sample_dist == "loguniform":
                low, high = np.log(self.temperature_dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == "beta":
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)
            # print("beta_enc", beta_enc)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {"beta": torch.tensor(beta), "encoding": beta_enc}

    def encode_conditional_information(self, steer_info: Tensor) -> Dict[str, Tensor]:
        n = len(steer_info)
        if self.temperature_sample_dist == "constant":
            beta = torch.ones(n) * self.temperature_dist_params
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            beta = torch.ones(n) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((n, self.num_thermometer_dim))

        assert len(beta.shape) == 1, f"beta should be of shape (Batch,), got: {beta.shape}"

        # TODO: positional assumption here, should have something cleaner
        preferences = steer_info[:, : len(self.objectives)].float()
        focus_dir = steer_info[:, len(self.objectives) :].float()

        preferences_enc, focus_enc = self.get_steer_encodings(preferences, focus_dir)
        encoding = torch.cat([beta_enc, preferences_enc, focus_enc], 1).float()

        return {
            "beta": beta,
            "encoding": encoding,
            "preferences": preferences,
            "focus_dir": focus_dir,
        }

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        # print("cond_info", cond_info)
        # print("flat_reward", flat_reward)
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log() #- original 
        # scalar_logreward = flat_reward.squeeze().clamp(min=1e-30) # - Kishan modified
        
        # print("scalar_logreward", flat_reward)
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info["beta"])

    def compute_flat_rewards(self, xml_robots, graphs, timesteps, env_id) -> Tuple[FlatRewards, Tensor]:
        eprewmeans = []
        # start_time = time.time()
        write_robots_to_file(xml_robots)
        # print("POLICY_PATH file path", POLICY_PATH)
        call_train_script(POLICY_PATH, timesteps, env_id)
        
        no_return_value = 0.001
        valid_robots = []

        for robot in xml_robots:
            try:
                eprewmean = modify_and_read_file(POLICY_PATH, robot)
                if eprewmean is None:
                    print("None as eprewmean for robot:", robot)
                    eprewmeans.append(no_return_value)
                    valid_robots.append(False)
                else:
                    eprewmeans.append(eprewmean)
                    valid_robots.append(True)
            except Exception as inner_e:
                print(f"Error processing robot {robot}: {inner_e}")
                eprewmeans.append(no_return_value)
                valid_robots.append(False)

        eprewmeans = list(np.around(np.array(eprewmeans),1))
        print("eprewmeans", eprewmeans)

        valid_eprewmeans = [e for e, v in zip(eprewmeans, valid_robots) if v]
        # print("valid_eprewmeans", valid_eprewmeans)
        mean = np.mean(valid_eprewmeans) if valid_eprewmeans else 0
        median = np.median(valid_eprewmeans) if valid_eprewmeans else 0
        std = np.std(valid_eprewmeans) if valid_eprewmeans else 0
        print("Max value & mean, median, std.dev", str(max(valid_eprewmeans)) if valid_eprewmeans else "N/A", mean, median, std)

        wandb.log({"mean": mean, "median": median, "Std.Dev": std})

        is_valid = torch.tensor(valid_robots).bool()
        # print("is_valid", is_valid)
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        eprewmeans_arr = np.array(eprewmeans)
        preds = self.flat_reward_transform(eprewmeans_arr).clip(1e-4, 1000).reshape((-1, 1))     #changed clip value from 100 to 1000 - Kishan
        return FlatRewards(preds), is_valid
        

class RoboTrainer(GFNTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "bootstrap_own_reward": False,
            "learning_rate": 1e-4,
            "Z_learning_rate": 1e-3,
            "global_batch_size": 25,
            "num_emb": 256,
            "num_layers": 4,
            "tb_epsilon": None,
            "tb_p_b_is_parameterized": False,
            "illegal_action_logreward": -75,
            "reward_loss_multiplier": 1, 
            "temperature_sample_dist": "uniform",
            "temperature_dist_params": (0.0, 64.0),
            "weight_decay": 1e-8,
            "num_data_loader_workers": 8,
            "momentum": 0.9,                            #High momentum - changed to 0.2 from 0.9 - kishan
            "adam_eps": 1e-8,
            "lr_decay": 20000,            
            "Z_lr_decay": 50000,
            "clip_grad_type": "norm",
            "clip_grad_param": 10,
            "random_action_prob": 0,                     # originally - 0.01, changed later to 0.05 and ran 25k horizon exp. increased to 0.1 for 100k horizon
            "valid_random_action_prob": 0.0,
            "sampling_tau": 0.0,
            "max_nodes": 9,
            "num_thermometer_dim": 32,
            "use_replay_buffer": True,           #used to be False, changed to True - Kishan
            "replay_buffer_size": 10000,         # changed from 10_000 to 5_000
            "replay_buffer_warmup": 10000,
            "mp_pickle_messages": False,
            "algo": "TB",
            "num_final_gen_steps": 1
        }

    def setup_algo(self):
        algo = self.hps.get("algo", "TB")
        if algo == "TB":
            algo = TrajectoryBalance
        elif algo == "FM":
            algo = FlowMatching
        else:
            raise ValueError(algo)
        self.algo = algo(self.env, self.ctx, self.rng, self.hps, max_nodes=self.hps["max_nodes"])

    def setup_task(self):
        self.task = RoboGenTask(
            dataset=self.training_data,
            temperature_distribution=self.hps["temperature_sample_dist"],
            temperature_parameters=self.hps["temperature_dist_params"],
            rng=self.rng,
            num_thermometer_dim=self.hps["num_thermometer_dim"],
            wrap_model=self._wrap_for_mp,
            log_dir=self.hps["log_dir"]
        )

    def setup_model(self):
        self.model = GraphTransformerGFN(self.ctx, log_dir=self.hps["log_dir"], data_collection_iters= self.hps["init_data_iters"], num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"])

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.hps["max_nodes"], num_cond_dim=self.hps["num_thermometer_dim"]
        )

    def setup(self):
        hps = self.hps
        # RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(self.hps["seed"])
        self.env = GraphBuildingEnv()
        self.training_data = []
        self.test_data = []
        self.offline_ratio = 0
        self.valid_offline_ratio = 0
        self.replay_buffer = (
            ReplayBuffer(self.hps["replay_buffer_size"], self.hps["replay_buffer_warmup"], self.rng)
            if self.hps["use_replay_buffer"]
            else None
        )
        self.setup_env_context()
        self.setup_algo()
        self.setup_task()
        self.setup_model()

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(
            non_Z_params,
            hps["learning_rate"],
            (hps["momentum"], 0.999),
            weight_decay=hps["weight_decay"],
            eps=hps["adam_eps"],
        )
        self.opt_Z = torch.optim.Adam(Z_params, hps["Z_learning_rate"], (0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / hps["lr_decay"]))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2 ** (-steps / hps["Z_lr_decay"]))

        self.sampling_tau = hps["sampling_tau"]
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model
        eps = hps["tb_epsilon"]
        hps["tb_epsilon"] = ast.literal_eval(eps) if isinstance(eps, str) else eps

        self.mb_size = hps["global_batch_size"]
        self.clip_grad_param = hps["clip_grad_param"]
        self.clip_grad_callback = {
            "value": (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            "norm": (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            "none": (lambda x: None),
        }[hps["clip_grad_type"]]

        os.makedirs(self.hps["log_dir"], exist_ok=True)
        fmt_hps = "\n".join([f"{f'{k}':40}:\t{f'({type(v).__name__})':10}\t{v}" for k, v in sorted(self.hps.items())])
        print(f"\n\nHyperparameters:\n{'-'*50}\n{fmt_hps}\n{'-'*50}\n\n")
        with open(pathlib.Path(self.hps["log_dir"]) / "hps.json", "w") as f:
            json.dump(self.hps, f)

    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))
    
    def update_max_nodes(self, iteration):
        # Calculate new max_nodes value based on iteration
        if iteration < 300:
            new_max_nodes = np.random.randint(3, self.hps["max_nodes"] + 1)  # Random integer between 3 and 10 (inclusive)
        else:
            new_max_nodes = self.hps["max_nodes"]    
        # Update max_nodes in hps
        # self.hps["max_nodes"] = new_max_nodes
        
        # Update algo
        self.algo.max_nodes = new_max_nodes
        
        # Update env context
        self.ctx.max_frags = new_max_nodes
        
        # Update model if necessary
        # Depending on your GraphTransformerGFN implementation, you might need to update it here
        
        print(f"Updated max_nodes to {new_max_nodes} at iteration {iteration}")

    def run(self, logger=None):

        if logger is None:
            logger = create_logger(logfile=self.hps["log_dir"] + "/train.log")

        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 500)             # changed from 1 to 20 - Kishan
        valid_freq = self.hps.get("validate_every", 0)
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.hps.get("checkpoint_every", valid_freq)
        traindl_start_time = time.time()
        train_dl = self.build_training_data_loader()   
        print("---Built training data loader in %s seconds ---" % (time.time() - traindl_start_time))
        # valddl_start_time = time.time()
        valid_dl = self.build_validation_data_loader()
        # print("---Built validation data loader in %s seconds ---" % (time.time() - valddl_start_time))

        if self.hps.get("num_final_gen_steps", 0) > 0:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.hps.get("start_at_step", 0) + 1
        logger.info("Starting training")

        train_start_time = time.time()
        train_dl_iter = cycle(train_dl)
        for it in range(start, 1 + self.hps["num_training_steps"]):
            self.update_max_nodes(it)
            batch = next(train_dl_iter)
            #write iteration count to file
            with open(os.path.join(self.hps["log_dir"], 'counter.txt'), 'w') as f:
                f.write(f"{it}")

            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            start = time.time()
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            # print("---Train time in %s seconds ---" % (time.time() - start))

            self.log(info, it, "train")
            if it % self.print_every == 0:
                logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

            if valid_freq > 0 and it % valid_freq == 0:
                for batch in valid_dl:
                    start_eval = time.time()
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    # print("---Eval time in %s seconds ---" % (time.time() - start_eval))
                    self.log(info, it, "valid")
                    logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")
            if ckpt_freq > 0 and it % ckpt_freq == 0:
                # print("Saving Checkpoint")
                self._save_state(it)
        self._save_state(self.hps["num_training_steps"])
        # print("---Overall (train + validation time in %s seconds ---" % (time.time() - train_start_time))
        num_final_gen_steps = self.hps.get("num_final_gen_steps", 0)
        if num_final_gen_steps > 0:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                range(self.hps["num_training_steps"], self.hps["num_training_steps"] + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                pass
            logger.info("Final generation steps completed.")


def main():
    global POLICY_PATH
    global EXP_METHOD

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument("--run_path", default="./logs")
    parser.add_argument("--base_xml_path", type=str, default='./assets/base_ant_incline.xml')
    parser.add_argument("--env_id", type=str, default='Ant-v5')
    parser.add_argument("--start_point", type=str, default='base')
    parser.add_argument("--env_terrain", type=str, default='wall')
    parser.add_argument("--terrain_from_external_source", type=int, default=1)
    parser.add_argument("--name", help='experiment_name', type=str, default='test')
    parser.add_argument("--exp_method", help='experiment method', type=str, default='naive')
    parser.add_argument("--rl_timesteps", help='rl_timesteps', type=int, default=4_000)
    parser.add_argument("--min_steps", help='min steps for gsca', type=int, default=30000)
    parser.add_argument("--max_gfn_nodes", help='graph nodes', type=int, default=10)
    parser.add_argument("--lastbatch_rl_timesteps", help='lastbatch_rl_timesteps', type=int, default=1_000_000)

    args = parser.parse_args()

    EXP_METHOD = args.exp_method

    args.terrain_from_external_source = bool(args.terrain_from_external_source)

    if args.terrain_from_external_source:
        parser.add_argument("--terrain_path", type=str, default=f"./assets/{args.env_terrain}_terrain.png")
        args = parser.parse_args()

    wandb.init(
    # Set the project where this run will be logged
    project="robonet_tests", 
    name=f"{args.name}",
    notes="RBN+mujoco+sb3",
    mode="online",  # "disabled" or "online"
    tags=["1k iter", "beta_decay", "randacts0"]
    )


    print("Env:", args.base_xml_path)
    # Step 1: Create a new folder
    postfix = int(time.time())
    log_folder = "exp_{}_{}_{}".format(args.name, args.seed, postfix)
    log_folder_path = os.path.join(args.run_path, log_folder)
    os.mkdir(log_folder_path)

    # Step 2: Copy the xml file to this folder
    xml_folder = "xmlrobots"
    xml_folder_path = os.path.join(log_folder_path, xml_folder)
    os.mkdir(xml_folder_path)
    shutil.copy(args.base_xml_path, os.path.join(os.path.abspath(xml_folder_path), "env.xml"))
    if args.terrain_from_external_source:
        shutil.copy(args.terrain_path, os.path.join(os.path.abspath(xml_folder_path), f"{args.env_terrain}_terrain.png"))
        tree = ET.parse(os.path.join(os.path.abspath(xml_folder_path), "env.xml"))
        root = tree.getroot()
        # Find the hfield element and update the file attribute
        for hfield in root.iter('hfield'):
            hfield.set('file', (os.path.join(os.path.abspath(xml_folder_path), f"{args.env_terrain}_terrain.png")))

        # Save the updated XML file
        tree.write(os.path.join(os.path.abspath(xml_folder_path), "env.xml"))

    xml_path = os.path.join(os.path.abspath(xml_folder_path))
    
    policy_folder = "policies"
    POLICY_PATH = os.path.join(log_folder_path, policy_folder)
    os.mkdir(POLICY_PATH)

    videos_folder = "videos"
    videos_folder_path = os.path.join(log_folder_path, videos_folder)
    os.mkdir(videos_folder_path)      

    hps = {
        "log_dir": f"{log_folder_path}",
        "xml_path": f"{xml_path}",
        "gum_beta": 0.0,
        "overwrite_existing_exp": True,
        "qm9_h5_path": "/data/chem/qm9/qm9.h5",
        "num_training_steps": 1000,
        "validate_every": 250,
        "lr_decay": 20000,
        "sampling_tau": 0.99,
        "num_data_loader_workers": 0,
        "temperature_dist_params": (0.0, 64.0),
        "max_nodes": args.max_gfn_nodes,
        "rl_timesteps": f"{args.rl_timesteps}",
        "lastbatch_rl_timesteps":f"{args.lastbatch_rl_timesteps}",
        "env":f"{args.env}",
        "start_point":f"{args.start_point}",    
        "exp_method":f"{args.exp_method}",    
        "env_terrain":f"{args.env_terrain}",
        "resource_per_link":f"{args.min_steps}",
        "init_data_iters": 10,
        "seed": int(f"{args.seed}"),
        "offline_ratio": 0.5,
        "env_id": f"{args.env_id}"
    }
    
    trial = RoboTrainer(hps, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()


