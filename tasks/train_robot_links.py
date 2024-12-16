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
from torch import Tensor
from torch.utils.data import Dataset
import wandb

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

def cycle(it):
    while True:
        for i in it:
            yield i


class RoboGenTask(GFNTask):
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
            assert isinstance(self.temperature_dist_params, float)
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
            elif self.temperature_sample_dist == "loguniform":
                low, high = np.log(self.temperature_dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == "beta":
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)

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
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        # scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log() #- original 
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30) # - Kishan modified
        
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info["beta"])

    def compute_flat_rewards(self, xml_robots, graphs, timesteps, env_id) -> Tuple[FlatRewards, Tensor]:
        pred_rews = []
        start_time = time.time()

        optim = 5       #choosen
        max_rew = 100
        
        for each_graph in graphs:
            # print("nodes in graph", len(each_graph.nodes))
            if len(each_graph.nodes) != optim:
                reward = max_rew - 4*abs(len(each_graph.nodes) - optim)                     # 50 nodes multiple used is 2, now chnaged to 20 for 9 nodes - Kishan
            elif len(each_graph.nodes) == optim:
                reward = max_rew
            else:
                reward = 0.001
            pred_rews.append(reward*4)

        print("pred_rews", pred_rews)
        
        is_valid = torch.tensor([i is not None for i in xml_robots]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        pred_rews_arr = np.array(pred_rews)
        preds = self.flat_reward_transform(pred_rews_arr).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid

class RoboTrainer(GFNTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "bootstrap_own_reward": False,
            "learning_rate": 1e-4,
            "Z_learning_rate": 1e-3,
            "num_emb": 128,
            "num_layers": 3,
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
            "random_action_prob": 0.3,                     # originally - 0.01, changed later to 0.05 and ran 25k horizon exp. increased to 0.1 for 100k horizon
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
        self.model = GraphTransformerGFN(self.ctx, num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"])

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.hps["max_nodes"], num_cond_dim=self.hps["num_thermometer_dim"]
        )

    def setup(self):
        hps = self.hps
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
        if iteration < 150:
            base_max_nodes = self.hps["max_nodes"] + 1
            new_max_nodes = 3 + (iteration % (base_max_nodes - 3))
        else:
            new_max_nodes = self.hps["max_nodes"] 
        
        self.algo.max_nodes = new_max_nodes
        self.ctx.max_frags = new_max_nodes
        
        print(f"Updated max_nodes to {new_max_nodes} at iteration {iteration}")

    def run(self, logger=None):
        if logger is None:
            logger = create_logger(logfile=self.hps["log_dir"] + "/train.log")

        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 50)
        valid_freq = self.hps.get("validate_every", 0)
        ckpt_freq = self.hps.get("checkpoint_every", valid_freq)
        train_dl = self.build_training_data_loader()   
        valid_dl = self.build_validation_data_loader()

        if self.hps.get("num_final_gen_steps", 0) > 0:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.hps.get("start_at_step", 0) + 1
        logger.info("Starting training")

        train_dl_iter = cycle(train_dl)
        for it in range(start, 1 + self.hps["num_training_steps"]):
            self.update_max_nodes(it)
            batch = next(train_dl_iter)
            with open(os.path.join(self.hps["log_dir"], 'counter.txt'), 'w') as f:
                f.write(f"{it}")

            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)

            self.log(info, it, "train")
            if it % self.print_every == 0:
                logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

            if valid_freq > 0 and it % valid_freq == 0:
                for batch in valid_dl:
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, "valid")
                    logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")
            if ckpt_freq > 0 and it % ckpt_freq == 0:
                self._save_state(it)
        self._save_state(self.hps["num_training_steps"])
        
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
    parser.add_argument("--env", type=str, default='ant')
    parser.add_argument("--start_point", type=str, default='base')
    parser.add_argument("--env_terrain", type=str, default='incline')
    parser.add_argument("--terrain_from_external_source", type=int, default=0)
    parser.add_argument("--name", help='experiment_name', type=str, default='test')
    parser.add_argument("--exp_method", help='experiment method', type=str, default='naive')
    parser.add_argument("--rl_timesteps", help='rl_timesteps', type=int, default=4_000)
    parser.add_argument("--min_steps", help='min steps for gsca', type=int, default=30000)
    parser.add_argument("--max_gfn_nodes", help='graph nodes', type=int, default=10)
    parser.add_argument("--lastbatch_rl_timesteps", help='lastbatch_rl_timesteps', type=int, default=1_000_000)
    parser.add_argument("--global_batch_size", help='batch size per iteration', default=32)
    parser.add_argument("--offline_data_iters", help='iterations count for offline data collection', type=int, default=150)


    args = parser.parse_args()
    
    postfix = int(time.time())
    log_folder = "exp_{}_{}_{}".format(args.name, args.seed, postfix)
    log_folder_path = os.path.join(args.run_path, log_folder)
    os.mkdir(log_folder_path)

    xml_folder = "xmlrobots"
    xml_folder_path = os.path.join(log_folder_path, xml_folder)
    os.mkdir(xml_folder_path)
    shutil.copy(args.base_xml_path, os.path.join(os.path.abspath(xml_folder_path), "env.xml"))

    xml_path = os.path.join(os.path.abspath(xml_folder_path))
    
    policy_folder = "policies"
    POLICY_PATH = os.path.join(log_folder_path, policy_folder)
    os.mkdir(POLICY_PATH)      

    hps = {
        "log_dir": f"{log_folder_path}",
        "xml_path": f"{xml_path}",
        "gum_beta": 0.0,
        "overwrite_existing_exp": True,
        "qm9_h5_path": "/data/chem/qm9/qm9.h5",
        "num_training_steps": 1000,
        "validate_every": 100,
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
        "epochs": 10,
        "global_batch_size": args.global_batch_size,
        "ctrl_cost_weight": 0.0005,
        "replay_buffer_size": int(args.offline_data_iters)*int(args.global_batch_size),
        "replay_buffer_warmup": int(args.offline_data_iters)*int(args.global_batch_size),
    }

    # print("32 bs, Nw size increased to 128x3, epochs = 50, changing maxnodes till 150 iterations. buffer size = 10000. Action size - 25x4, rew*4 and log removed")
    
    trial = RoboTrainer(hps, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    trial.print_every = 1
    trial.run()

if __name__ == "__main__":
    main()
