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
import git
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
import math
import xml.etree.ElementTree as ET
# wandb.login()
# wandb.init(
#     # Set the project where this run will be logged
#     project="test", 
#     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#     name="Ant_links_test", 
#     )
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



POLICY_PATH = ""

def calculate_depth(graph, node, visited, depth):
    visited[node] = True  # Mark the current node as visited
    max_depth = depth  # Initialize the current depth

    # Iterate through the neighbors of the current node
    for neighbor in graph[node]:
        if not visited[neighbor]:
            # Recursively calculate the depth of the neighbor
            max_depth = max(max_depth, calculate_depth(graph, neighbor, visited, depth + 1))

    return max_depth

def graph_depth(nodes, edges):
    # Create an adjacency list representation of the graph
    graph = {node: [] for node in nodes}
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])  # Uncomment if the graph is undirected

    # Initialize visited flags for nodes
    visited = {node: False for node in nodes}

    max_depth = 0
    # Perform DFS from each unvisited node to find the maximum depth
    for node in nodes:
        if not visited[node]:
            max_depth = max(max_depth, calculate_depth(graph, node, visited, 0))

        return max_depth

def sigmoid(d, k=1, c=2):
        return 1 / (1 + math.exp(-k * (d - c)))

# Function to calculate the magnitude of a vector
def calculate_vector_length(fromto_values):
    vec1 = np.array([float(fromto_values[0]), float(fromto_values[1]), float(fromto_values[2])])
    vec2 = np.array([float(fromto_values[3]), float(fromto_values[4]), float(fromto_values[5])])
    # The length of the body is the magnitude of the difference between these vectors
    length = np.linalg.norm(vec2 - vec1)
    return length

# Recursive function to traverse the body tree and find the longest branch
def find_longest_branch_length(body):
    # print("body", body.attrib['name'])
    max_branch_length = 0.0  # To store the maximum length of any branch
    current_length = 0.0  # To store the length of the current body

    # Check if this body has a 'geom' element with a 'fromto' attribute
    geom = body.find('geom')
    if geom is not None and 'fromto' in geom.attrib:
        fromto_values = geom.attrib['fromto'].split()
        current_length = calculate_vector_length(fromto_values)
        # print("current_length", current_length)

    # Recursively find the longest branch among all child bodies
    for child_body in body.findall('body'):
        child_branch_length = find_longest_branch_length(child_body)
        # print("child_branch_length", child_branch_length)
        max_branch_length = max(max_branch_length, child_branch_length)

    # The total length for this body is its own length plus the longest of its children
    return current_length + max_branch_length

class SEHTask(GFNTask):
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

    # def _load_task_models(self):
    #     model = bengio2021flow.load_original_model()
    #     model, self.device = self._wrap_model(model, send_to_device=True)
    #     return {"seh": model}

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
        """
        Encode conditional information at validation-time
        We use the maximum temperature beta for inference
        Args:
            steer_info: Tensor of shape (Batch, 2 * n_objectives) containing the preferences and focus_dirs
            in that order
        Returns:
            Dict[str, Tensor]: Dictionary containing the encoded conditional information
        """
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
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log()  # - original 
        # scalar_logreward = flat_reward.squeeze().clamp(min=1e-30) # - Kishan modified
        
        # print("scalar_logreward", flat_reward)
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info["beta"])

    # multiprocessing with rew from mujoco simulations
    def compute_flat_rewards(self, xml_robots, graphs, rl_timesteps) -> Tuple[FlatRewards, Tensor]:

        start = time.time()
        
        pred_rews = []
        min_rew = 10
        for robot in xml_robots:
            tree = ET.parse(robot)
            root = tree.getroot()
            root_body = root.find('.//body')
            robot_length = find_longest_branch_length(root_body)
            pred_rews.append(robot_length * min_rew)

        print("pred_rews", pred_rews, len(pred_rews))
        print("mean & std", np.mean(pred_rews), np.std(pred_rews))
        print("max value", str(max(pred_rews)))

        wandb.log({"mean": np.mean(pred_rews), "Std.Dev": np.std(pred_rews)})

        end = time.time()

        # print("time taken for coputing rews", end-start)

        is_valid = torch.tensor([i is not None for i in xml_robots]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        pred_rews_arr = np.array(pred_rews)
        preds = self.flat_reward_transform(pred_rews_arr).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid
        

class SEHFragTrainer(GFNTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "bootstrap_own_reward": False,
            "learning_rate": 1e-4,
            "Z_learning_rate": 1e-3,
            "global_batch_size": 32,
            "num_emb": 64,
            "num_layers": 2,
            "tb_epsilon": None,
            "tb_p_b_is_parameterized": False,
            "illegal_action_logreward": -75,
            "reward_loss_multiplier": 1,
            "temperature_sample_dist": "uniform",
            "temperature_dist_params": (0.5, 32.0),
            "weight_decay": 1e-8,
            "num_data_loader_workers": 8,
            "momentum": 0.9,                            #High momentum - changed to 0.2 from 0.9 - kishan
            "adam_eps": 1e-8,
            "lr_decay": 20000,
            "Z_lr_decay": 50000,
            "clip_grad_type": "norm",
            "clip_grad_param": 10,
            "random_action_prob": 0.01,
            "valid_random_action_prob": 0.0,
            "sampling_tau": 0.0,
            "max_nodes": 9,
            "num_thermometer_dim": 32,
            "use_replay_buffer": False,
            "replay_buffer_size": 10000,
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
        self.task = SEHTask(
            dataset=self.training_data,
            temperature_distribution=self.hps["temperature_sample_dist"],
            temperature_parameters=self.hps["temperature_dist_params"],
            rng=self.rng,
            num_thermometer_dim=self.hps["num_thermometer_dim"],
            wrap_model=self._wrap_for_mp,
            log_dir=self.hps["log_dir"]
        )

    def setup_model(self):
        self.model = GraphTransformerGFN(self.ctx, num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"])

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.hps["max_nodes"], num_cond_dim=self.hps["num_thermometer_dim"]
        )

    def setup(self):
        hps = self.hps
        # RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(14285)
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

        # saving hyperparameters
        # git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        # self.hps["gflownet_git_hash"] = git_hash

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


def main():


    global POLICY_PATH

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument("--run_path", default="./logs")
    parser.add_argument("--base_xml_path", default="./assets/base_ant_flat.xml")
    parser.add_argument("--name", help='experiment_name', type=str)

    args = parser.parse_args()


    wandb.init(
    # Set the project where this run will be logged
    project="gfn_toyproj", 
    name=f"{args.name}",
    notes="gfn_design",
    mode="online",  # "disabled" or "online"
    )

    
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

    xml_path = os.path.join(os.path.abspath(xml_folder_path))
    
    policy_folder = "policies"
    POLICY_PATH = os.path.join(log_folder_path, policy_folder)
    os.mkdir(POLICY_PATH)      

    hps = {
        "log_dir": f"./{log_folder_path}",
        "xml_path": f"{xml_path}",
        "overwrite_existing_exp": True,
        "qm9_h5_path": "/data/chem/qm9/qm9.h5",
        "num_training_steps": 1000,
        "validate_every": 25,
        "lr_decay": 20000,
        "sampling_tau": 0.99,
        "num_data_loader_workers": 0,
        "temperature_dist_params": (0.0, 64.0),
        "max_nodes": 6,
        "rl_timesteps": 0,
        "exp_method": "toy-rew-func"
    }

    trial = SEHFragTrainer(hps, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # num_cpu_cores = os.cpu_count()
    # print("num_cpu_cores", num_cpu_cores)

    # num_available_processors = multiprocessing.cpu_count()
    # print("Number of available processors:", num_available_processors)

    trial.print_every = 1
    trial.run()

if __name__ == "__main__":
    main()
