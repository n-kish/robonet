import copy
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from pprint import pprint
import os
# import wandb
from envs.graph_building_env import GraphAction, GraphActionType
import envs.robot_env 


def beta_decay(iteration, beta_init):
    iter_window = 150
    decay_step_size = 120
    if iteration > iter_window:
        alpha = 0
    else: 
        alpha = 1

    return (alpha * beta_init) - (alpha * beta_init * 0.5**((iteration - iter_window) / decay_step_size)) + (beta_init * 0.5**((iteration - iter_window) / decay_step_size))


class GraphSampler:
    """A helper class to sample from GraphActionCategorical-producing models"""

    def __init__(
        self, ctx, env, max_len, max_nodes, rng, sample_temp=1, correct_idempotent=False, pad_with_terminal_state=False
    ):
        """
        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        rng: np.random.RandomState
            rng used to take random actions
        sample_temp: float
            [Experimental] Softmax temperature used when sampling
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        pad_with_terminal_state: bool
            [Experimental] If true pads trajectories with a terminal
        """
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        self.rng = rng
        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

    def sample_from_model(
        self, model: nn.Module, xml_dir: str, log_dir:str, is_final_sample, n: int, exp_method, min_resource, cond_info: Tensor, dev: torch.device, env_id: str, random_action_prob: float = 0.0):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """

        self.min_resource = min_resource
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for i in range(n)]

        graphs = [self.env.new() for i in range(n)]
        # print("GRAPHS", graphs[1], len(graphs))
        done = [False] * n

        # TODO: instead of padding with Stop, we could have a virtual action whose probability
        # always evaluates to 1. Presently, Stop should convert to a [0,0,0] aidx, which should
        # always be at least a valid index, and will be masked out anyways -- but this isn't ideal.
        # Here we have to pad the backward actions with something, since the backward actions are
        # evaluated at s_{t+1} not s_t.
        bck_a = [[GraphAction(GraphActionType.Stop)] for i in range(n)]
        # print("bck_a", bck_a[1], len(bck_a))

        def not_done(lst):
            # print("testing if done")
            return [e for i, e in enumerate(lst) if not done[i]]
        
        counter_k = 0
        # print("MAX_LEN", self.max_len)
        
        #Reading iteration number from file to determine gumbel beta value 
        count_dir = os.path.dirname(log_dir)
        try:
            with open(os.path.join(count_dir, 'counter.txt'), 'r') as counter_file:
                iteration = int(counter_file.read())
                # print("iteration", iteration)
        except FileNotFoundError:
            iteration = 0
            print("iteration from exception", iteration)
        except Exception as e:
            print("An error occurred:", e)

        if not is_final_sample:
            beta_init = 10
            gum_beta = beta_decay(iteration, beta_init)
        else:
            gum_beta = 0.1
        
        print("gum_beta", gum_beta)

 
        for t in range(self.max_len):
            # print("t in loop", t)
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
            # TODO: compute bck_cat.log_prob(bck_a) when relevant
            fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            # tmp = vars(fwd_cat)
            # pprint(tmp)

            if random_action_prob > 0:
                # print("Entering")
                masks = [1] * len(fwd_cat.logits) if fwd_cat.masks is None else fwd_cat.masks
                # print("mask", masks)
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev
                ).float()
                # print("is random action", is_random_action)
                # Set the logits to some large value if they're not masked, this way the masked
                # actions have no probability of getting sampled, and there is a uniform
                # distribution over the rest
                fwd_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                    for i, m, b in zip(fwd_cat.logits, masks, fwd_cat.batch)
                ]
                # print("fwd_cat.logits", fwd_cat.logits)

                
            if self.sample_temp != 1:
                sample_cat = copy.copy(fwd_cat)
                sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
                actions = sample_cat.sample(gum_beta)
                # print("if actions", actions)
            else:
                # print("fwd_cat.logits", len(fwd_cat.logits[0]))
                actions = fwd_cat.sample(gum_beta)
                # print("actions", actions)
                # actions_exp = fwd_cat.sample(0.1)
                # actions[0] = actions_exp[0]

                # print("else actions", actions, len(actions))
                # exit()
            # print("fwd_cat", fwd_cat)
            graph_actions = [self.ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            # print("graph_actions", graph_actions, len(graph_actions))
            # print("grah_action0", graph_actions[0], graph_actions[0].action)
            # print("Value, source, target", graph_actions[0].value, graph_actions[0].source, graph_actions[0].target)
            # counter_k += 1
            # if counter_k == 10:
            #     exit()
            # print("graph_actions", graph_actions, len(graph_actions)) # 64 graph actions
            log_probs = fwd_cat.log_prob(actions)
            # print("log_probs",fwd_cat.log_prob(actions)) # 64 log_probs of actions
            # Step each trajectory, and accumulate statistics
            # print("Collected graph_actions")
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                bck_a[i].append(self.env.reverse(graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    # print("received STOP ACTION")
                    # check for min condition and only allow stop after min is crossed - kishan
                    # print("graphs[i].number_of_nodes()", graphs[i].number_of_nodes())
                    # if graphs[i].number_of_nodes() > 4: 
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                    # Cannot use this following code block yet because graph sizes being produced are only of 2 nodes. 
                    # else:
                    #     counter = 0
                    #     while counter < 5 and graph_actions[j].action is GraphActionType.Stop:
                    #         counter = counter + 1
                    #         action = fwd_cat.sample()
                    #         print("action", action)
                    #         graph_action = self.ctx.aidx_to_GraphAction(graphs[i], action[i])
                    #         print("graph action", graph_action)
                    #         graph_actions[j].action = graph_action.action
                    #         data[i]["traj"].append((graphs[i], graph_actions[j]))
                    #         bck_a[i].append(self.env.reverse(graphs[i], graph_actions[j]))
                        
                    #     gp = graphs[i]
                    #     try:
                    #         # self.env.step can raise AssertionError if the action is illegal
                    #         gp = self.env.step(graphs[i], graph_actions[j])
                    #         assert len(gp.nodes) <= self.max_nodes
                    #     except AssertionError:
                    #         done[i] = True
                    #         data[i]["is_valid"] = False
                    #         bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    #         data[i]["is_sink"].append(1)
                    #         continue
                    #     if t == self.max_len - 1:
                    #         done[i] = True
                    #     # If no error, add to the trajectory
                    #     # P_B = uniform backward
                    #     n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    #     bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    #     data[i]["is_sink"].append(0)
                    #     graphs[i] = gp
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]["is_valid"] = False
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        data[i]["is_sink"].append(1)
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    data[i]["is_sink"].append(0)
                    graphs[i] = gp
                    # print("graphs from l208 in graph_sampling.py", graphs)
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(env_id, graphs[i], xml_dir, log_dir, exp_method, self.min_resource):
                    # check if the graph is sane (e.g. RDKit can
                    # construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]["is_valid"] = False
            if all(done):
                break
        # print("t_value", t)

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1

        # There are 3 types of possible trajectories
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop)], P_B = [..., bck(gp), 1]
        #  B - ends with an invalid action.  = [..., (g, a)],                 = [..., 1]
        #  C - ends at max_len.              = [..., (g, a)],                 = [..., bck(gp)]

        # Let's say we pad terminal states, then:
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop), (gp, None)], P_B = [..., bck(gp), 1, 1]
        #  B - ends with an invalid action.  = [..., (g, a), (g, None)],                  = [..., 1, 1]
        #  C - ends at max_len.              = [..., (g, a), (gp, None)],                 = [..., bck(gp), 1]
        # and then P_F(terminal) "must" be 1

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                # TODO: instead of padding with Stop, we could have a virtual action whose
                # probability always evaluates to 1.
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data
