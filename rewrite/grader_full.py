'''
GRADER (GRAph-based Discovery and Exploration for Reinforcement learning)
完整实现，包含所有必要组件
'''

import os
import sys
import copy
import math
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import re
import pickle as pkl
import pandas as pd
import scipy.stats as stats
import gym
from gym import spaces
from gym.utils import seeding
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import skimage
import skimage.draw

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.utils import data

# 尝试导入fcit，如果没有，提供安装建议
try:
    from fcit import fcit
except ImportError:
    print("ERROR: fcit module not found. Please install it using: pip install fcit")
    print("fcit is used for fast conditional independence testing")
    print("More info: https://github.com/kjchalup/fcit")
    print("This is a required dependency for GRADER to function properly.")
    # 如果是关键依赖，可以在这里抛出异常中断程序
    raise

# ---------------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------------

def CPU(var):
    return var.detach().cpu().numpy()


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def load_config(config_path="config.yml"):
    """
    加载配置文件
    """
    default_config = {
        'GRADER': {
            'pretrain_buffer_size': 1000,
            'max_buffer_size': 10000, 
            'epsilon': 0.1,
            'model_path': './models',
            'model_id': 0,
            'lr': 0.001,
            'batch_size': 64,
            'hidden_dim': 64,
            'hidden_size': 256,
            'n_epochs': 10,
            'validation_flag': True,
            'validation_freq': 5,
            'validation_ratio': 0.2,
            'planner': {
                'type': 'random',
                'horizon': 5,
                'gamma': 0.99,
                'popsize': 500
            },
            'discover': {
                'discovery_interval': 10
            }
        },
        'SAC': {
            'pretrain_buffer_size': 1000,
            'max_buffer_size': 10000,
            'min_Val': 1e-7,
            'batch_size': 64, 
            'update_iteration': 10,
            'gamma': 0.99,
            'tau': 0.01,
            'alpha': 0.2,
            'hidden_dim': 256,
            'lr': 0.0003,
            'model_path': './models',
            'model_id': 0
        }
    }
    
    try:
        import yaml
        if os.path.isfile(config_path):
            f = open(config_path)
            return yaml.load(f, Loader=yaml.FullLoader)
        else:
            print(f"Configuration file not found at: {config_path}")
            print("Using default configuration")
            return default_config
    except ImportError:
        print("yaml module not found. Please install it using: pip install pyyaml")
        print("Using default configuration")
        return default_config
    

def get_colors_and_weights(cmap='Set1', num_colors=9, observed=True, mode='Train', new_colors=None):
    """Get color array from matplotlib colormap."""
    EPS = 1e-17
    
    def observed_colors(num_colors, mode):
        if mode == 'ZeroShot':
            c = np.sort(np.random.uniform(0.0, 1.0, size=num_colors))
        else:
            c = (np.arange(num_colors)) / (num_colors-1)
            diff = 1.0 / (num_colors - 1)
            if mode == 'Train':
                diff = diff / 8.0
            elif mode == 'Test-v1':
                diff = diff / 4.0
            elif mode == 'Test-v2':
                diff = diff / 3.0
            elif mode == 'Test-v3':
                diff = diff / 2.0

            unif = np.random.uniform(-diff+EPS, diff-EPS, size=num_colors)
            unif[0] = abs(unif[0])
            unif[-1] = -abs(unif[-1])

            c = c + unif

        return c
    
    def get_cmap(cmap, mode):
        length = 9
        if cmap == 'Sets':
            if "FewShot" not in mode:
                cmap = plt.get_cmap('Set1')
            else:
                cmap = [plt.get_cmap('Set1'), plt.get_cmap('Set3')]
                length = [9,12]
        else:
            if "FewShot" not in mode:
                cmap = plt.get_cmap('Pastel1')
            else:
                cmap = [plt.get_cmap('Pastel1'), plt.get_cmap('Pastel2')]
                length = [9,8]

        return cmap, length
    
    def unobserved_colors(cmap, num_colors, mode, new_colors=None):
        if mode in ['Train', 'ZeroShotShape']:
            cm, length = get_cmap(cmap, mode)
            weights = np.sort(np.random.choice(length, num_colors, replace=False))
            colors = [cm(i/length) for i in weights]
        else:
            cm, length = get_cmap(cmap, mode)
            cm1, cm2 = cm
            length1, length2 = length
            l = length1 + len(new_colors)
            w = np.sort(np.random.choice(l, num_colors, replace=False))
            colors = []
            weights = []
            for i in w:
                if i < length1:
                    colors.append(cm1(i/length1))
                    weights.append(i)
                else:
                    colors.append(cm2(new_colors[i - length1] / length2))
                    weights.append(new_colors[i - length1] + 0.5)

        return colors, weights
    
    if observed:
        c = observed_colors(num_colors, mode)
        cm = plt.get_cmap(cmap)

        colors = []
        for i in reversed(range(num_colors)):
            colors.append((cm(c[i])))

        weights = [num_colors - idx for idx in range(num_colors)]
    else:
        colors, weights = unobserved_colors(cmap, num_colors, mode, new_colors)
    return colors, weights


# ---------------------------------------------------------------------------------
# GNN Modules
# ---------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_input=7, n_output=6, n_h=1, size_h=128, dropout_p=0.0):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, size_h)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_list = nn.ModuleList()
        for _ in range(n_h):
            self.fc_list.append(nn.Linear(size_h, size_h))
        self.fc_out = nn.Linear(size_h, n_output)

    def forward(self, x):
        x = x.view(-1, self.n_input)
        out = self.fc_in(x)
        out = self.dropout(out)
        out = self.relu(out)
        for layer in self.fc_list:
            out = layer(out)
            out = self.dropout(out)
            out = self.relu(out)
        out = self.fc_out(out)
        return out


class IndividualEmbedding(nn.Module):
    def __init__(self, in_features, out_features, node_num, bias=True):
        super(IndividualEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(node_num, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(node_num, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        # x - [B, node_num, in_features]
        # W - [node_num, in_features, out_features]
        # output - [B,, node_num, out_features]
        output = []
        for n_i in range(x.shape[1]):
            o_i = torch.matmul(x[:, n_i, :], self.weight[n_i])
            if self.bias is not None:
                o_i += self.bias[n_i]
            o_i = o_i[:, None, :]
            output.append(o_i)

        output = torch.cat(output, dim=1)
        return output


class RelationGraphConvolution(nn.Module):
    """
    Relation GCN layer. 
    """
    def __init__(self, in_features, out_features, edge_dim, aggregate='mean', dropout=0., use_relu=False, bias=False):
        """
        Args:
            in_features: scalar of channels for node embedding
            out_features: scalar of channels for node embedding
            edge_dim: dim of edge type, virtual type not included
        """
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.aggregate = aggregate
        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = None

        self.weight = nn.Parameter(torch.FloatTensor(self.edge_dim, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.edge_dim, 1, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, adj):
        """
        Args:
            x: (B, max_node_num, node_dim): 
            adj: (B, edge_dim, max_node_num, max_node_num): 

        Returns:
            node_embedding: (B, max_node_num, embed_size): updated embedding for nodes x
        """
        x = F.dropout(x, p=self.dropout, training=self.training)  # (B, max_node_num, node_dim)

        # transform
        support = torch.einsum('bid, edh-> beih', x, self.weight) # (B, edge_dim, max_node_num, embed_size)

        # works as a GCN with sum aggregation
        output = torch.einsum('beij, bejh-> beih', adj, support)  # (B, edge_dim, max_node_num, embed_size)

        if self.bias is not None:
            output += self.bias
        if self.act is not None:
            output = self.act(output)  # (B, E, N, d)

        if self.aggregate == 'sum':
            # sum pooling #(b, N, d)
            node_embedding = torch.sum(output, dim=1, keepdim=False)
        elif self.aggregate == 'max':
            # max pooling  #(b, N, d)
            node_embedding = torch.max(output, dim=1, keepdim=False)
        elif self.aggregate == 'mean':
            # mean pooling #(b, N, d)
            node_embedding = torch.mean(output, dim=1, keepdim=False)
        else:
            print('GCN aggregate error!')
        return node_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGCN(nn.Module):
    def __init__(self, node_dim, node_num, aggregate, hidden_dim, output_dim, edge_dim, hidden_num, dropout=0.0, bias=True):
        """
        Args:
            node_dim:
            hidden_dim:
            output_dim:
            edge_dim:
            num_layars: the number of layers in each R-GCN
            dropout:
        """
        super(RGCN, self).__init__()
        self.hidden_num = hidden_num

        self.emb = nn.Linear(node_dim, hidden_dim, bias=bias) 
        self.ind_emb = IndividualEmbedding(node_dim, hidden_dim, node_num=node_num, bias=bias) 

        self.gc1 = RelationGraphConvolution(hidden_dim, hidden_dim, edge_dim=edge_dim, aggregate=aggregate, use_relu=True, dropout=dropout, bias=bias)
        self.gc2 = nn.ModuleList([RelationGraphConvolution(hidden_dim, hidden_dim, edge_dim=edge_dim, aggregate=aggregate, use_relu=True, dropout=dropout, bias=bias) for i in range(hidden_num)])
        self.gc3 = RelationGraphConvolution(hidden_dim, output_dim, edge_dim=edge_dim, aggregate=aggregate, use_relu=False, dropout=dropout, bias=bias)

    def forward(self, x, adj):
        # embedding layer (individual for each node)
        x = self.ind_emb(x)

        # first GCN layer
        x = self.gc1(x, adj)

        # hidden GCN layer(s)
        for i in range(self.hidden_num):
            x = self.gc2[i](x, adj)  # (#node, #class)

        # last GCN layer
        x = self.gc3(x, adj)  # (batch, N, d)

        # return node embedding
        return x


class GRU_SCM(nn.Module):
    def __init__(self, action_dim_list, state_dim_list, node_num, aggregate, hidden_dim, edge_dim, hidden_num, dropout=0.0, bias=True, random=False):
        super(GRU_SCM, self).__init__()
        self.random = random
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.node_num = node_num

        # NOTE: assume we dont input the padding zero, just use seperate embedding layers with different size of input dimensions
        self.node_list = action_dim_list + state_dim_list
        self.relu = nn.ReLU()
        self.input_embs = []
        for i in self.node_list:
            self.input_embs.append(nn.Linear(i, hidden_dim))
        self.input_embs = nn.ModuleList(self.input_embs)

        self.output_embs = []
        for i in self.node_list:
            self.output_embs.append(nn.Linear(hidden_dim, i))
        self.output_embs = nn.ModuleList(self.output_embs)

        self.gru_list = []
        for i in range(len(self.node_list)):
            self.gru_list.append(nn.GRU(hidden_dim, hidden_dim, num_layers=self.hidden_num, batch_first=True))
        self.gru_list = nn.ModuleList(self.gru_list)

    def forward(self, x_in, adj):
        # x - [B, N, d] - [B, A+S, d]
        # adj - [B, E, N, N]

        # diagnal is not necessary for aggregation
        adj = adj[0, 0] - torch.eye(x_in.shape[1], device=adj.device)

        # extract the true nodes then pass them throught the emdeddings
        x_list = []
        for e_i, embd_i in enumerate(self.input_embs):
            x_i = embd_i(x_in[:, e_i, 0:self.node_list[e_i]])[:, None, :]
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        x = self.relu(x)

        agg_list = []
        for n_i in range(x.shape[1]):
            hidden = x[:, n_i:n_i+1, :].transpose(0, 1).contiguous()

            # add noise U to the hidden variable
            if self.random:
                noise = (torch.rand(hidden.shape, device=hidden.device) - 0.5) * 0.1
                hidden = hidden + noise

            # NOTE: GRU requires the order to be fixed, while attention is permuational
            neighbors_idx = torch.nonzero(adj[n_i], as_tuple=False)[:, 0]
            if len(neighbors_idx) == 0:
                # if there is no neighbor, use the embedding itself
                agg_list.append(hidden.transpose(0, 1))
            else:
                neighbors = x[:, neighbors_idx, :].contiguous()
                aggregation, _ = self.gru_list[n_i](neighbors, hidden)
                agg_list.append(aggregation[:, -1:, :])
        x = torch.cat(agg_list, dim=1)
        x = self.relu(x)

        # convert the embedding back to the true node for loss calculation
        x_padded = torch.zeros_like(x_in) # [B, N, d]
        for e_i, embd_i in enumerate(self.output_embs):
            x_i = embd_i(x[:, e_i, :])
            x_padded[:, e_i, 0:self.node_list[e_i]] = x_i
        return x_padded


# ---------------------------------------------------------------------------------
# Random Shooting MPC
# ---------------------------------------------------------------------------------

class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class RandomOptimizer(Optimizer):
    def __init__(self, action_dim, horizon, popsize):
        super().__init__()
        self.horizon = horizon
        self.popsize = popsize
        self.action_dim = action_dim
        self.solution = None
        self.cost_function = None

    def setup(self, cost_function):
        self.cost_function = cost_function

    def reset(self):
        pass

    def generate_one_action(self, low, high, size):
        shape = torch.Size(size)
        if torch.cuda.is_available():
            move = torch.cuda.LongTensor(shape)
        else:
            move = torch.LongTensor(shape)

        torch.randint(0, high, size=shape, out=move)
        move = torch.nn.functional.one_hot(move)
        return move

    def obtain_solution_chemistry(self, action_dim):     
        # convert int to onehot
        action = np.random.randint(0, action_dim, size=(self.popsize, self.horizon))
        action = (np.arange(action_dim) == action[..., None]).astype(int)
        costs = self.cost_function(action)
        solution = action[np.argmin(costs)]
        return solution


class MPC_Chemistry(object):
    def __init__(self, mpc_args):
        self.type = mpc_args['type']
        self.horizon = mpc_args['horizon']
        self.gamma = mpc_args['gamma']
        self.popsize = mpc_args['popsize']

        # parameters from the environment
        self.action_dim = mpc_args['env_params']['action_dim']
        self.goal_dim = mpc_args['env_params']['goal_dim']

        self.optimizer = RandomOptimizer(action_dim=self.action_dim, horizon=self.horizon, popsize=self.popsize)
        self.optimizer.setup(self.cost_function)
        self.reset()

    def reset(self):
        self.optimizer.reset()

    def act(self, model, state):
        # process the state to get pure state and goal
        goal = state[len(state)-self.goal_dim:]
        pure_state = state[:len(state)-self.goal_dim] # remove the goal info at very beginning

        self.model = model
        self.state = pure_state
        self.goal = state = np.repeat(goal[None], self.popsize, axis=0)

        best_solution = self.optimizer.obtain_solution_chemistry(self.action_dim)

        # task the first step as our action
        action = best_solution[0]
        return action

    def preprocess(self, state):
        state = np.repeat(self.state[None], self.popsize, axis=0)
        return state

    def cost_function(self, actions):
        # the observation need to be processed since we use a common model
        state = self.preprocess(self.state)
        stop_flag = np.ones(self.popsize,)

        assert actions.shape == (self.popsize, self.horizon, self.action_dim)
        costs = np.zeros(self.popsize)
        for t_i in range(self.horizon):
            action = actions[:, t_i, :]  # (batch_size, timestep, action dim)
            # the output of the prediction model is [state_next - state]
            state_next = self.model.predict(state, action) + state
            cost, stop_mask = self.chemistry_objective(state_next)  # compute cost
            stop_flag = stop_flag * stop_mask # Bit AND, stopped trajectory will have 0 cost
            costs += (1-stop_flag) * cost
            state = copy.deepcopy(state_next)

        return costs

    def chemistry_objective(self, state):
        mse = np.sum((state - self.goal) ** 2, axis=1) ** 0.5
        final_cost = mse
        stop_mask = mse < 0.1 # goal achieved
        return final_cost, stop_mask


# ---------------------------------------------------------------------------------
# Causal Discovery
# ---------------------------------------------------------------------------------

class Discover(object):
    def __init__(self, args):
        self.goal_dim = args['env_params']['goal_dim']
        self.action_dim = args['env_params']['action_dim']
        self.env_name = args['env_params']['env_name']

        # parameters
        self.num_perm = 8
        self.prop_test = 0.5

        if self.env_name == 'chemistry':
            self.pvalue_threshold = 0.01
            self.num_objects = args['env_params']['num_objects']
            self.num_colors = args['env_params']['num_colors']
            self.width = args['env_params']['width']
            self.height = args['env_params']['height']
            self.adjacency_matrix = args['env_params']['adjacency_matrix']
            self.state_dim_list = [self.num_colors * self.width * self.height] * self.num_objects 
            self.action_dim_list = [self.num_objects * self.num_colors] # action does not have causal variables
            self.adj_node_num = len(self.action_dim_list) + len(self.state_dim_list) 
            self.state_dim_list = self.state_dim_list * 2 
            self.ground_truth = self.adjacency_matrix + np.eye(self.adjacency_matrix.shape[0]) # add diagonal elements

            # nodes (Action x1, state x20)
            if self.num_objects == 10:
                self.next_state_offset = 11
                self.node_name_mapping = {
                    0: 'A_i',  
                    1: 'S_0',  2: 'S_1', 3: 'S_2', 4: 'S_3', 5: 'S_4', 6: 'S_5', 7: 'S_6', 8: 'S_7', 9: 'S_8', 10: 'S_9',
                    11: 'NS_0', 12: 'NS_1', 13: 'NS_2', 14: 'NS_3', 15: 'NS_4', 16: 'NS_5', 17: 'NS_6', 18: 'NS_7', 19: 'NS_8', 20: 'NS_9'
                }
            elif self.num_objects == 5:
                self.next_state_offset = 6
                self.node_name_mapping = {
                    0: 'A_i',  
                    1: 'S_0',  2: 'S_1', 3: 'S_2', 4: 'S_3', 5: 'S_4',
                    6: 'NS_0', 7: 'NS_1', 8: 'NS_2', 9: 'NS_3', 10: 'NS_4'
                }

            # remove the dimension that has no influence
            self.remove_list = [[] for _ in self.node_name_mapping.keys()]
            # variable type is discrete or not
            self.discrete_var = {
                0: False, 
                1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False, 10: False, 
                11: False, 12: False, 13: False, 14: False, 15: False, 16: False, 17: False, 18: False, 19: False, 20: False
            }
        else:
            raise ValueError('unknown env name')

        # build the variable list
        self.node_dim_list = self.action_dim_list + self.state_dim_list 
        self.var_list = [self.node_name_mapping[n_i] for n_i in range(len(self.node_name_mapping.keys()))]
        self.intervene_var_list = self.var_list.copy()

        # build causal graph
        self.reset_causal_graph()

        # build dataset 
        self.dataset_dict = {i: [] for i in range(len(self.node_dim_list))}

    def reset_causal_graph(self):
        self.causal_graph = nx.DiGraph()
        # add nodes to causal graph
        for var_i in self.var_list:
            self.causal_graph.add_node(var_i) 

    def load_data(self, path, used_ratio):
        if used_ratio > 1:
            raise ValueError('used_ratio should be smaller than 1')

        self.dataset_dict = np.load(path, allow_pickle=True).item()
        data_size = len(self.dataset_dict[0])
        print('loaded data size:', data_size)
        used_size = int(used_ratio * data_size)
        for k_i in self.dataset_dict.keys():
            self.dataset_dict[k_i] = self.dataset_dict[k_i][0:used_size]

    def store_transition(self, data):
        # [state, action, next_state]
        # we should remove the goal infomation from x and label
        state = data[0][:len(data[0])-self.goal_dim]
        action = data[1]
        next_state = data[2][:len(data[0])-self.goal_dim]
        delta_state = next_state - state

        if self.env_name == 'chemistry':
            # check whether the intervention is valid
            action_check = np.argmax(action)
            # only change color of one object one time
            obj_id = action_check // self.num_colors
            color_id = action_check % self.num_colors 
            state_check = state.reshape(self.num_objects, self.num_colors, self.width, self.height)
            state_check = state_check.sum(3)
            state_check = state_check.sum(2)
            if state_check[obj_id][color_id] == 1: # the intervention will not have influence
                return 

        #state_next_state = np.concatenate([state, next_state])
        state_next_state = np.concatenate([state, delta_state], axis=0)

        # build the nodes of action
        start_ = 0
        for a_i in range(len(self.action_dim_list)):
            end_ = self.action_dim_list[a_i] + start_
            node_a = action[start_:end_]
            self.dataset_dict[a_i].append(node_a)
            start_ = end_

        # build the nodes of state
        start_ = 0
        for s_i in range(len(self.state_dim_list)):
            end_ = self.state_dim_list[s_i] + start_
            node_s = state_next_state[start_:end_] 

            if self.env_name == 'chemistry':
                # remove position
                node_s = node_s.reshape(self.num_colors, self.width, self.height)
                node_s = np.sum(node_s, axis=2)
                node_s = np.sum(node_s, axis=1)

            self.dataset_dict[s_i+len(self.action_dim_list)].append(node_s)
            start_ = end_

    def _two_variable_test(self, i, j, cond_list):
        # get x variable
        x = copy.deepcopy(np.array(self.dataset_dict[i]))
        x = np.delete(x, self.remove_list[i], axis=1)
        name_x = self.node_name_mapping[i]

        # get y variable
        y = copy.deepcopy(np.array(self.dataset_dict[j]))
        y = np.delete(y, self.remove_list[j], axis=1)
        name_y = self.node_name_mapping[j]

        # independency test
        if len(cond_list) == 0:
            pvalue = fcit.test(x, y, z=None, num_perm=self.num_perm, prop_test=self.prop_test, discrete=(self.discrete_var[i], self.discrete_var[j]))
        # conditional independency test
        else:
            z = []
            for z_idx in cond_list:
                z_i = copy.deepcopy(np.array(self.dataset_dict[z_idx]))
                z_i = np.delete(z_i, self.remove_list[z_idx], axis=1)
                z.append(z_i)
            z = np.concatenate(z, axis=1)
            pvalue = fcit.test(x, y, z=z, num_perm=self.num_perm, prop_test=self.prop_test, discrete=(self.discrete_var[i], self.discrete_var[j]))

        name_z = ''
        for k in cond_list:
            name_z += self.node_name_mapping[k] 
            name_z += ' '

        #print(name_x, 'and', name_y, 'condition on [', name_z, '] , pvalue is {:.5f}'.format(pvalue))
        return pvalue

    def _two_variable_test_chisquare(self, i, j):
        name_x = self.node_name_mapping[i]
        name_y = self.node_name_mapping[j]
        contingency_table = pd.crosstab(self.dataframe[name_y], self.dataframe[name_x])

        # a table summarization of two categorical variables in this form is called a contingency table.
        _, pvalue, _, _ = stats.chi2_contingency(contingency_table)
        #print(name_x, 'and', name_y, 'pvalue is {:.5f}'.format(pvalue))
        return pvalue

    def _is_action(self, name):
        return True if name.split('_')[0] == 'A' else False

    def _is_state(self, name):
        return True if name.split('_')[0] == 'S' else False

    def _is_next(self, name):
        return True if name.split('_')[0] == 'NS' else False

    def select_action(env, state):
        ''' For interventional discovery, actively select action. For random discovery, randomly select actions '''
        action = env.random_action()
        return action

    def update_causal_graph(self):
        # convert the dataset dict to dataframe for discrete variables
        if self.env_name in ['chemistry']:
            data_dict = {}
            for n_i in self.dataset_dict.keys():
                x = copy.deepcopy(np.array(self.dataset_dict[n_i]))
                x = np.delete(x, self.remove_list[n_i], axis=1)
                name_x = self.node_name_mapping[n_i]
                x_str = list(map(np.array2string, list(x)))
                data_dict[name_x] = x_str
            self.dataframe = pd.DataFrame(data_dict)

        # start the test
        for i in range(len(self.node_dim_list)):
            for j in range(len(self.node_dim_list)):
                name_i = self.node_name_mapping[i]
                name_j = self.node_name_mapping[j]

                # directly add edges from S_xx to NS_xx
                if self._is_state(name_i) and self._is_next(name_j) and name_i.split('_')[1] == name_j.split('_')[1]:
                    self.causal_graph.add_edge(name_i, name_j)

                # for chemistry env, the causal direction will be lower triangular matrix
                if self.env_name in ['chemistry']:
                    if self._is_state(name_i) and self._is_next(name_j) and name_i.split('_')[1] > name_j.split('_')[1]:
                        continue

                action_state = self._is_action(name_i) and self._is_next(name_j)
                state_state = self._is_state(name_i) and self._is_next(name_j) and name_i.split('_')[1] != name_j.split('_')[1]
                if not action_state and not state_state:
                    continue

                # do independent test
                p_value = self._two_variable_test_chisquare(i, j)
                if p_value < self.pvalue_threshold:
                    self.causal_graph.add_edge(name_i, name_j)

        # visualize graph
        #self.visualize_graph(self.causal_graph, './log/causal_graph.png', directed=True)

    def get_true_causal_graph(self):
        if self.env_name == 'chemistry':
            truth_graph = nx.DiGraph()
            # add action edges
            for i in range(0, self.ground_truth.shape[0]):
                truth_graph.add_edge(self.node_name_mapping[0], self.node_name_mapping[i+self.next_state_offset])
            # add state edges
            for i in range(self.ground_truth.shape[0]):
                for j in range(self.ground_truth.shape[1]):
                    if self.ground_truth[j, i] == 1: # need to transpose
                        truth_graph.add_edge(self.node_name_mapping[i+1], self.node_name_mapping[j+self.next_state_offset])
        else:
            raise ValueError('Unknown Environment Name')

        return truth_graph

    def _retrieve_adjacency_matrix(self, graph, order_nodes=None, weight=False):
        """Retrieve the adjacency matrix from the nx.DiGraph or numpy array."""
        if isinstance(graph, np.ndarray):
            return graph
        elif isinstance(graph, nx.DiGraph):
            if order_nodes is None:
                order_nodes = graph.nodes()
            if not weight:
                return np.array(nx.adjacency_matrix(graph, order_nodes, weight=None).todense())
            else:
                return np.array(nx.adjacency_matrix(graph, order_nodes).todense())
        else:
            raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")
        
    def SHD(self, target, pred, double_for_anticausal=True):
        ''' Reference: https://github.com/ElementAI/causal_discovery_toolbox/blob/master/cdt/metrics.py '''
        true_labels = self._retrieve_adjacency_matrix(target)
        predictions = self._retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

        diff = np.abs(true_labels - predictions)
        if double_for_anticausal:
            return np.sum(diff)
        else:
            diff = diff + diff.transpose()
            diff[diff > 1] = 1  # Ignoring the double edges.
            return np.sum(diff)/2

    def visualize_graph(self, causal_graph, save_path=None, directed=True):
        plt.figure(figsize=(4, 7))

        left_node = []
        node_color = []
        for n_i in causal_graph.nodes:
            if self._is_action(n_i) or self._is_state(n_i):
                left_node.append(n_i)
            
            if self._is_action(n_i):
                node_color.append('#DA87B3')
            elif self._is_state(n_i):
                node_color.append('#86A8E7')
            else:
                node_color.append('#56D1C9')

        pos = nx.bipartite_layout(causal_graph, left_node)
        nx.draw_networkx(causal_graph, pos, node_color=node_color, arrows=directed, with_labels=True, node_size=1400, arrowsize=20)

        plt.axis('off')
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300)
            plt.close('all')

    def get_adj_matrix_graph(self):
        # NOTE: discovered graph contains 2*N_S+N_A nodes, we need to convert it to N_S+N_A nodes
        node_mapping = {  
            'A_i': 0,                  
            'S_0': 1, 'S_1': 2, 'S_2': 3, 'S_3': 4, 'S_4': 5,
            'NS_0': 1, 'NS_1': 2, 'NS_2': 3, 'NS_3': 4, 'NS_4': 5,
        }
        adj_matrix = np.zeros((self.adj_node_num, self.adj_node_num))
        adj_matrix[0, 0] = 1
        edges = self.causal_graph.edges
        for e_i in edges:
            src_idx = node_mapping[e_i[0]]
            tar_idx = node_mapping[e_i[1]]
            adj_matrix[tar_idx, src_idx] = 1
        return adj_matrix

    def save_model(self, model_path, model_id):
        states = {'graph': self.causal_graph}
        filepath = os.path.join(model_path, 'graph.'+str(model_id)+'.pkl')
        with open(filepath, 'wb') as f:
            pkl.dump(states, f)

    def load_model(self, model_path, model_id):
        filepath = os.path.join(model_path, 'graph.'+str(model_id)+'.pkl')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = pkl.load(f)
            self.causal_graph = checkpoint['graph']
        else:
            raise Exception('No graph found!')


# ---------------------------------------------------------------------------------
# World Model and Planner
# ---------------------------------------------------------------------------------

class WorldModel(object):
    def __init__(self, args):
        self.state_dim = args['env_params']['state_dim']
        self.action_dim = args['env_params']['action_dim']
        self.goal_dim = args['env_params']['goal_dim']
        self.env_name = args['env_params']['env_name']
        self.grader_model = args['grader_model']
        self.use_discover = args['use_discover']
        self.use_gt = args['use_gt']

        assert self.grader_model in ['causal', 'full', 'mlp', 'offline', 'gnn']

        self.n_epochs = args['n_epochs']
        self.lr = args['lr']
        self.batch_size = args['batch_size']

        self.validation_flag = args['validation_flag']
        self.validate_freq = args['validation_freq']
        self.validation_ratio = args['validation_ratio']
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # process things that are different in environments
        if self.env_name == 'chemistry':
            self.build_node_and_edge = self.build_node_and_edge_chemistry
            self.organize_nodes = self.organize_nodes_chemistry
            self.num_objects = args['env_params']['num_objects']
            self.num_colors = args['env_params']['num_colors']
            self.width = args['env_params']['width']
            self.height = args['env_params']['height']
            self.adjacency_matrix = args['env_params']['adjacency_matrix']
            self.adjacency_matrix += np.eye(self.adjacency_matrix.shape[0]) # add diagonal elements
            self.state_dim_list = [self.num_colors * self.width * self.height] * self.num_objects 
            self.action_dim_list = [self.num_objects * self.num_colors] # action does not have causal variables
        else:
            raise ValueError('Unknown environment name')

        self.use_full = False
        self.use_mlp = False
        if self.grader_model == 'mlp':
            self.model_name = 'mlp'
            self.use_mlp = True
        elif self.grader_model == 'causal':
            self.model_name = 'gru'
        elif self.grader_model == 'full':
            self.model_name = 'gru'
            self.use_full = True
        elif self.grader_model == 'gnn':
            self.model_name = 'gnn'
            self.use_full = True

        random = False
        if self.model_name == 'mlp':
            input_dim = self.state_dim - self.goal_dim + self.action_dim
            output_dim = self.state_dim - self.goal_dim
            self.model = CUDA(MLP(input_dim, output_dim, args["hidden_dim"], args["hidden_size"], dropout_p=0.0))
            hidden_dim = args["hidden_size"]
        elif self.model_name == 'gru' or self.model_name == 'gnn':
            edge_dim = 1
            hidden_num = 1
            if self.env_name == 'chemistry':
                args["hidden_dim"] = 64
                
            hidden_dim = args["hidden_dim"]
            self.node_num = len(self.action_dim_list) + len(self.state_dim_list)
            self.node_dim = int(np.max(self.state_dim_list+self.action_dim_list))
            if self.model_name == 'gnn':
                self.model = CUDA(RGCN(self.node_dim, self.node_num, 'mean', args["hidden_dim"], self.node_dim, edge_dim, hidden_num))
            else:
                self.model = CUDA(GRU_SCM(self.action_dim_list, self.state_dim_list, self.node_num, 'mean', args["hidden_dim"], edge_dim, hidden_num, dropout=0.0, random=random))

        print('----------------------------')
        print('Env:', self.env_name)
        print('GRADER model:', self.grader_model)
        print('Model_name:', self.model_name)
        print('Full:', self.use_full)
        print('SCM noise:', random)
        print('Hidden dim:', hidden_dim)
        print('----------------------------')

        self.model.apply(kaiming_init)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.buffer_length = 0
        self.criterion = self.mse_loss

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.data = None
        self.label = None
        self.eps = 1e-30

        if self.grader_model == 'causal':
            # the initial graph is a lower triangular graph
            self.causal_graph = np.zeros((self.node_num, self.node_num))
            for i in range(self.causal_graph.shape[0]):
                for j in range(self.causal_graph.shape[1]):
                    if i >= j:
                        self.causal_graph[i, j] = 1
        self.best_test_loss = np.inf

    def build_node_and_edge_chemistry(self, data):
        # create the node matrix. the last node is the output node therefore should always be 0.
        batch_size = data.shape[0]
        x = torch.zeros((batch_size, self.node_num, self.node_dim), device=torch.device(self.device)) # [B, 125]

        # build the nodes of action
        action = data[:, sum(self.state_dim_list):]
        start_ = 0
        for a_i in range(len(self.action_dim_list)):
            end_ = self.action_dim_list[a_i] + start_
            x[:, a_i, 0:end_-start_] = action[:, start_:end_] # pad 0 for remaining places
            start_ = end_

        # build the nodes of state
        state = data[:, 0:sum(self.state_dim_list)]

        # [B, N*C*W*H] -> [B, N, C*W*H]
        state = state.reshape(batch_size, self.num_objects * self.num_colors, self.width, self.height)
        state = state.reshape(batch_size, self.num_objects, self.num_colors * self.width * self.height)
        start_ = 0
        for s_i in range(len(self.state_dim_list)):
            end_ = self.state_dim_list[s_i] + start_
            x[:, s_i+len(self.action_dim_list), 0:end_-start_] = state[:, s_i, :] # pad 0 for remaining places
            start_ = end_

        if self.use_full:
            # full graph (states are fully connected)
            full = np.ones((self.node_num, self.node_num))
            action_row = np.zeros((1, self.node_num))
            action_row[0] = 1
            full[0, :] = action_row
            adj = full

        if self.use_discover:
            adj = self.causal_graph

        if self.use_gt:
            # using GT causal graph
            gt_adj = np.zeros((self.node_num, self.node_num))
            gt_adj[1:, 1:] = self.adjacency_matrix
            gt_adj[:, 0] = 1.0
            adj = gt_adj

        adj = np.array(adj)[None, None, :, :].repeat(batch_size, axis=0)
        adj = CUDA(torch.from_numpy(adj.astype(np.float32)))
        return x, adj

    def organize_nodes_chemistry(self, x):
        # x - [B, node_num, node_dim], the nodes of next_state are in the end
        delta_state_node = x[:, -len(self.state_dim_list):, :]
        delta_state = []
        for s_i in range(len(self.state_dim_list)):
            state_i = delta_state_node[:, s_i:s_i+1, 0:self.state_dim_list[s_i]] 
            delta_state.append(state_i)

        # NOTE: since the embedding of state has beed reordered, we should do that thing again
        delta_state = torch.cat(delta_state, dim=1) # [B, N, C*W*H]
        delta_state = delta_state.reshape(delta_state.shape[0], self.num_objects * self.num_colors * self.width * self.height)
        return delta_state

    def data_process(self, data, max_buffer_size):
        x = data[0][None]
        label = data[1][None]
        self.buffer_length += 1
    
        # add new data point to data buffer
        if self.data is None:
            self.data = CUDA(torch.from_numpy(x.astype(np.float32)))
            self.label = CUDA(torch.from_numpy(label.astype(np.float32)))
        else:
            if self.data.shape[0] < max_buffer_size:
                self.data = torch.cat((self.data, CUDA(torch.from_numpy(x.astype(np.float32)))), dim=0)
                self.label = torch.cat((self.label, CUDA(torch.from_numpy(label.astype(np.float32)))), dim=0)
            else:
                # replace the old buffer
                #index = self.buffer_length % max_buffer_size # sequentially replace buffer
                index = np.random.randint(0, max_buffer_size) # randomly replace buffer
                self.data[index] = CUDA(torch.from_numpy(x.astype(np.float32)))
                self.label[index] = CUDA(torch.from_numpy(label.astype(np.float32)))

    def split_train_validation(self):
        num_data = len(self.data)

        # use validation
        if self.validation_flag:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]

            train_set = [[self.data[idx], self.label[idx]] for idx in train_idx]
            test_set = [[self.data[idx], self.label[idx]] for idx in test_idx]

            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_set = [[self.data[idx], self.label[idx]] for idx in range(num_data)]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
        return train_loader, test_loader

    def fit(self):
        self.model.train()
        train_loader, test_loader = self.split_train_validation()

        self.best_test_loss = np.inf
        for epoch in range(self.n_epochs):
            for datas, labels in train_loader:
                self.optimizer.zero_grad()

                if self.use_mlp:
                    delta = self.model(datas)
                    loss = self.criterion(delta, labels)
                else:
                    x, adj = self.build_node_and_edge(datas)
                    x = self.model(x, adj)
                    delta = self.organize_nodes(x)
                    loss = self.criterion(delta, labels)
                loss.backward()
                self.optimizer.step()

            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                with torch.no_grad():
                    loss_test = self.validate_model(test_loader)
                if loss_test < self.best_test_loss:
                    self.best_test_loss = loss_test
                    self.best_model = copy.deepcopy(self.model.state_dict())

        # load the best model if we use validation
        if self.validation_flag:
            self.model.load_state_dict(self.best_model)
        return self.best_test_loss

    def validate_model(self, testloader):
        self.model.eval()
        loss_list = []
        for datas, labels in testloader:
            if self.use_mlp:
                delta = self.model(datas)
                loss = self.criterion(delta, labels)
            else:
                x, adj = self.build_node_and_edge(datas)
                x = self.model(x, adj)
                delta = self.organize_nodes(x)
                loss = self.criterion(delta, labels)

            loss_list.append(loss.item())
        self.model.train()
        return np.mean(loss_list)

    def predict(self, s, a):
        self.model.eval()
        # convert to torch format
        if isinstance(s, np.ndarray):
            s = CUDA(torch.from_numpy(s.astype(np.float32)))
        if isinstance(a, np.ndarray):
            a = CUDA(torch.from_numpy(a.astype(np.float32)))

        inputs = torch.cat((s, a), axis=1)

        with torch.no_grad():
            if self.use_mlp:
                delta = self.model(inputs)
            else:
                x, adj = self.build_node_and_edge(inputs)
                x = self.model(x, adj)
                delta = self.organize_nodes(x)

            delta = delta.cpu().detach().numpy()
        return delta

    def save_model(self, model_path, model_id):
        states = {'model': self.model.state_dict()}
        filepath = os.path.join(model_path, 'grade.'+str(model_id)+'.torch')
        with open(filepath, 'wb') as f:
            torch.save(states, f)

    def load_model(self, model_path, model_id):
        filepath = os.path.join(model_path, 'grade.'+str(model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise Exception('No GRADE model found!')


class Planner(object):
    def __init__(self, args):
        self.pretrain_buffer_size = args['pretrain_buffer_size']
        self.max_buffer_size = args['max_buffer_size']
        self.epsilon = args['epsilon']
        self.goal_dim = args['env_params']['goal_dim']

        args['mpc']['env_params'] = args['env_params']
        self.mpc_controller = MPC_Chemistry(args['mpc'])
        self.mpc_controller.reset()

        self.model = WorldModel(args)

    def select_action(self, env, state, deterministic):
        if self.model.data is None or self.model.data.shape[0] < self.pretrain_buffer_size:
            action = env.random_action()
        else:
            if np.random.uniform(0, 1) > self.epsilon or deterministic:
                action = self.mpc_controller.act(model=self.model, state=state)
            else:
                action = env.random_action()
        return action

    def store_transition(self, data):
        # [state, action, next_state]
        # we should remove the goal infomation from x and label
        pure_state = data[0][:len(data[0])-self.goal_dim]
        action = data[1]
        pure_next_state = data[2][:len(data[0])-self.goal_dim]
        x = np.concatenate([pure_state, action])
        label = pure_next_state - pure_state 
        self.model.data_process([x, label], self.max_buffer_size)

    def train(self):
        # when data has been collected enough, train model
        if self.model.data is None or self.model.data.shape[0] < self.pretrain_buffer_size:
            self.best_test_loss = 0
        else:
            self.best_test_loss = self.model.fit()

    def set_causal_graph(self, causal_graph):
        self.model.causal_graph = causal_graph

    def save_model(self, model_path, model_id):
        self.model.save_model(model_path, model_id)

    def load_model(self, model_path, model_id):
        self.model.load_model(model_path, model_id)


# ---------------------------------------------------------------------------------
# GRADER Main Agent
# ---------------------------------------------------------------------------------

class GRADER(object):
    name = 'GRADER'

    def __init__(self, args):
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        args['planner']['env_params'] = args['env_params']
        args['planner']['grader_model'] = args['grader_model']
        args['discover']['env_params'] = args['env_params']

        # use discovered graph or not. use gt if not use discovered graph
        self.use_discover = True
        
        # only use causal when we use causal model
        if args['grader_model'] != 'causal':
            self.use_discover = False

        args['planner']['use_discover'] = self.use_discover
        args['planner']['use_gt'] = not self.use_discover

        # two modules
        self.planner = Planner(args['planner'])
        self.discover = Discover(args['discover'])

        # decide the ratio between generation and discovery (generation is always longer)
        self.stage = 'generation'
        self.episode_counter = 0
        self.discovery_interval = args['discover']['discovery_interval']

    def stage_scheduler(self):
        if (self.episode_counter + 1) % self.discovery_interval == 0:
            self.stage = 'discovery'
        else:
            self.stage = 'generation'
        self.episode_counter += 1

    def select_action(self, env, state, deterministic):
        return self.planner.select_action(env, state, deterministic)

    def store_transition(self, data):
        self.planner.store_transition(data)
        self.discover.store_transition(data)

    def train(self):
        # discovery
        if self.stage == 'discovery' and self.use_discover:
            self.discover.update_causal_graph()
            self.planner.set_causal_graph(self.discover.get_adj_matrix_graph())

        # generation
        self.planner.train()

        # in the end, update the stage
        self.stage_scheduler()

    def save_model(self):
        self.planner.save_model(self.model_path, self.model_id)
        self.discover.save_model(self.model_path, self.model_id)

    def load_model(self):
        self.planner.load_model(self.model_path, self.model_id)
        self.discover.load_model(self.model_path, self.model_id)


# ---------------------------------------------------------------------------------
# SAC (Soft Actor-Critic) Agent
# ---------------------------------------------------------------------------------

class Actor_Discrete(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, min_val):
        super(Actor_Discrete, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.min_val = min_val
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def sample(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        z = action_probs == 0.0  # deal with situation of 0.0 probabilities because we can't do log 0
        z = z.float() * self.min_val
        action_log_probs = torch.log(action_probs + z)
        return action, action_probs, action_log_probs      

    def select_action(self, state, deterministic=False):
        action, action_probs, _ = self.sample(state)
        if deterministic:
            return torch.argmax(action_probs)
        else:
            return action


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_type):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_type = action_type

        if self.action_type == 'continuous':
            # for continuous Critic, input is state_dim+action_dim and output is 1
            self.fc1 = nn.Linear(state_dim+action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
            self.fc4 = nn.Linear(state_dim+action_dim, hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, hidden_dim)
            self.fc6 = nn.Linear(hidden_dim, 1)
        else:
            # for discrete Critic, input is state_dim and output is action_dim
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
            self.fc4 = nn.Linear(state_dim, hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, hidden_dim)
            self.fc6 = nn.Linear(hidden_dim, action_dim)
        
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, s, a):
        if self.action_type == 'continuous':
            x = torch.cat([s, a], dim=1) # combination s and a
        else:
            x = s
        
        # Q1
        q1 = self.relu(self.fc1(x))
        q1 = self.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2
        q2 = self.relu(self.fc4(x))
        q2 = self.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2


class ReplayBuffer():
    def __init__(self, memory_capacity, buffer_dim):
        self.buffer_dim = buffer_dim
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((self.memory_capacity, self.buffer_dim))
        self.memory_counter = 0
        self.memory_len = 0

    def push(self, data):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = data
        self.memory_counter += 1
        self.memory_len = min(self.memory_len+1, self.memory_capacity)

    def sample(self, batch_size):
        sample_index = np.random.randint(0, self.memory_len, size=batch_size)
        batch_memory = self.memory[sample_index, :]
        return batch_memory


class SAC():
    ''' SAC model with continuous and discrete action space '''
    name = 'SAC'

    def __init__(self, args):
        super(SAC, self).__init__()
        self.max_buffer_size = args['max_buffer_size']
        self.pretrain_buffer_size = args['pretrain_buffer_size']
        self.lr = args['lr']
        self.min_val = torch.tensor(args['min_Val']).float()
        self.batch_size = args['batch_size']
        self.update_iteration = args['update_iteration']
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        hidden_dim = args['hidden_dim']

        self.model_path = args['model_path']
        self.model_id = args['model_id']

        self.state_dim = args['env_params']['state_dim']
        self.action_dim = args['env_params']['action_dim']
        self.env_name = args['env_params']['env_name']

        if self.env_name == 'chemistry':
            self.action_type = 'discrete'
            self.num_objects = args['env_params']['num_objects']
            self.num_colors = args['env_params']['num_colors']
            self.action_dim = 1
            action_num = self.num_objects * self.num_colors
            self.policy = CUDA(Actor_Discrete(self.state_dim, action_num, hidden_dim, self.min_val))
            self.critic = CUDA(QNetwork(self.state_dim, action_num, hidden_dim, self.action_type))
            self.critic_target = CUDA(QNetwork(self.state_dim, action_num, hidden_dim, self.action_type))
            self._action_postprocess = self._chemistry_action_postprocess

            # build the mapping bettwen index to action
            self.action_map = []
            self.action_map_str = []
            for i in range(action_num):
                onehot = np.zeros((action_num,))
                onehot[i] = 1.0
                self.action_map.append(onehot)
                self.action_map_str.append(str(onehot))
        else:
            raise ValueError('Unknown env name')

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # buffer for saving data
        self.replay_buffer = ReplayBuffer(self.max_buffer_size, self.state_dim*2+self.action_dim+2)

    def _chemistry_action_postprocess(self, action_idx):
        return self.action_map[action_idx]

    def select_action(self, env, state, deterministic=False):
        state = CUDA(torch.from_numpy(state.astype(np.float32)))
        action = self.policy.select_action(state, deterministic)
        action = self._action_postprocess(CPU(action))
        return action

    def store_transition(self, data):
        # [state, action, reward, next_state, done]
        # for discrete action, we need to store the index
        if self.action_type == 'discrete':
            action = data[1]
            action_idx = self.action_map_str.index(str(action))
            data[1] = np.array([action_idx])

        data = np.concatenate([data[0], data[1], [data[2]], data[3], [np.float32(data[4])]])
        self.replay_buffer.push(data)

    def update_loss_discrete(self, state, action, reward, next_state, mask):
        # update critic
        with torch.no_grad():
            _, action_probs, action_log_probs = self.policy.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, None)
            next_q_value = action_probs * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * action_log_probs)
            next_q_value = reward + (mask * self.gamma * next_q_value.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        qf1, qf2 = self.critic(state, None)
        qf1 = qf1.gather(dim=1, index=action.long())
        qf2 = qf2.gather(dim=1, index=action.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute actor loss
        _, action_probs, action_log_probs = self.policy.sample(state)
        qf1, qf2 = self.critic(state, None)
        min_Q = torch.min(qf1, qf2)
        policy_loss = (action_probs * (self.alpha * action_log_probs - min_Q)).sum(1).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
    
    def update_loss_continuous(self, state, action, reward, next_state, mask):
        # update critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + mask * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # update policy network
        pi, log_pi, _ = self.policy.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

    def train(self):
        if self.replay_buffer.memory_len < self.pretrain_buffer_size:
            #print('Skip training, buffer size: [{}/{}]'.format(self.replay_buffer.memory_len, self.max_buffer_size))
            return

        for _ in range(self.update_iteration):
            # Sample replay buffer
            batch_memory = self.replay_buffer.sample(self.batch_size)
            state_batch = CUDA(torch.from_numpy(batch_memory[:, 0:self.state_dim].astype(np.float32)))
            action_batch = CUDA(torch.from_numpy(batch_memory[:, self.state_dim:self.state_dim+self.action_dim].astype(np.float32)))
            reward_batch = CUDA(torch.from_numpy(batch_memory[:, self.state_dim+self.action_dim:self.state_dim+self.action_dim+1].astype(np.float32)))
            next_state_batch = CUDA(torch.from_numpy(batch_memory[:, self.state_dim+self.action_dim+1:2*self.state_dim+self.action_dim+1].astype(np.float32)))
            mask_batch = CUDA(torch.from_numpy(1-batch_memory[:, 2*self.state_dim+self.action_dim+1:2*self.state_dim+self.action_dim+2].astype(np.float32)))

            if self.action_type == 'continuous':
                self.update_loss_continuous(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
            else:
                self.update_loss_discrete(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

    def save_model(self):
        states = {'policy': self.policy.state_dict(), 'critic': self.critic.state_dict(), 'critic_target': self.critic_target.state_dict()}
        filepath = os.path.join(self.model_path, 'model.sac.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.sac.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        else:
            raise Exception('No SAC model found!')


# ---------------------------------------------------------------------------------
# Chemistry Environment
# ---------------------------------------------------------------------------------

def parse_skeleton(graph, M=None):
    """
    Parse the skeleton of a causal graph in the mini-language of --graph.
    
    The mini-language is:
        
        GRAPH      = "" CHAIN{, CHAIN}*
        CHAIN      = INT_OR_SET {-> INT_OR_SET}
        INT_OR_SET = INT | SET
        INT        = [0-9]*
        SET        = \{ SET_ELEM {, SET_ELEM}* \}
        SET_ELEM   = INT | INT_RANGE
        INT_RANGE  = INT - INT
    """
    
    regex = re.compile(r'''
        \s*                                      # Skip preceding whitespace
        (                                        # The set of tokens we may capture, including
        [,]                                  | # Commas
        (?:\d+)                              | # Integers
        (?:                                    # Integer set:
            \{                                   #   Opening brace...
            \s*                                #   Whitespace...
            \d+\s*(?:-\s*\d+\s*)?              #   First integer (range) in set...
            (?:,\s*\d+\s*(?:-\s*\d+\s*)?\s*)*  #   Subsequent integers (ranges)
            \}                                   #   Closing brace...
        )                                    | # End of integer set.
        (?:->)                                 # Arrows
        )
    ''', re.A | re.X)
    
    # Utilities
    def parse_int(s):
        try:    
            return int(s.strip())
        except: 
            return None
    
    def parse_intrange(s):
        try:
            sa, sb = map(str.strip, s.strip().split("-", 1))
            sa, sb = int(sa), int(sb)
            sa, sb = min(sa,sb), max(sa,sb)+1
            return range(sa,sb)
        except:
            return None
    
    def parse_intset(s):
        try:
            i = set()
            for s in map(str.strip, s.strip()[1:-1].split(",")):
                if parse_int(s) is not None: 
                    i.add(parse_int(s))
                else:                        
                    i.update(set(parse_intrange(s)))
            return sorted(i)
        except:
            return None
    
    def parse_either(s):
        asint = parse_int(s)
        if asint is not None: return asint
        asset = parse_intset(s)
        if asset is not None: return asset
        raise ValueError
    
    def find_max(chains):
        m = 0
        for chain in chains:
            for link in chain:
                link = max(link) if isinstance(link, list) else link
                m = max(link, m)
        return m
    
    # Crack the string into a list of lists of (ints | lists of ints)
    graph  = [graph] if isinstance(graph, str) else graph
    chains = []
    for gstr in graph:
        for chain in re.findall("((?:[^,{]+|\{.*?\})+)+", gstr, re.A):
            links = list(map(str.strip, regex.findall(chain)))
            assert(len(links) & 1)
            
            chain = [parse_either(links.pop(0))]
            while links:
                assert links.pop(0) == "->"
                chain.append(parse_either(links.pop(0)))
            chains.append(chain)
    
    # Find the maximum integer referenced within the skeleton
    uM = find_max(chains)+1
    if M is None:
        M = uM
    else:
        assert(M >= uM)
        M = max(M, uM)
    
    # Allocate adjacency matrix.
    gamma = np.zeros((M,M), dtype=np.float32)
    
    # Interpret the skeleton
    for chain in chains:
        for prevlink, nextlink in zip(chain[:-1], chain[1:]):
            if isinstance(prevlink, list) and isinstance(nextlink, list):
                for i in nextlink:
                    for j in prevlink:
                        if i > j:
                            gamma[i,j] = 1
            elif isinstance(prevlink, list) and isinstance(nextlink, int):
                for j in prevlink:
                    if nextlink > j:
                        gamma[nextlink,j] = 1
            elif isinstance(prevlink, int) and isinstance(nextlink, list):
                minn = min(nextlink)
                if minn == prevlink:
                    raise ValueError("Edges are not allowed from " + str(prevlink) + " to oneself!")
                elif minn < prevlink:
                    raise ValueError("Edges are not allowed from " + str(prevlink) + " to ancestor " + str(minn) + " !")
                else:
                    for i in nextlink:
                        gamma[i,prevlink] = 1
            elif isinstance(prevlink, int) and isinstance(nextlink, int):
                if nextlink == prevlink:
                    raise ValueError("Edges are not allowed from " + str(prevlink) + " to oneself!")
                elif nextlink < prevlink:
                    raise ValueError("Edges are not allowed from " + str(prevlink) + " to ancestor " + str(nextlink) + " !")
                else:
                    gamma[nextlink,prevlink] = 1
    
    # Return adjacency matrix.
    return gamma


def random_dag(M, N, g=None):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    if g is None:
        expParents = 5
        idx = np.arange(M).astype(np.float32)[:,np.newaxis]
        idx_maxed = np.minimum(idx * 0.5, expParents)
        p = np.broadcast_to(idx_maxed/(idx+1), (M, M))
        B = np.random.binomial(1, p)
        B = np.tril(B, -1)
        return B
    else:
        gammagt = parse_skeleton(g, M=M)
        return gammagt


graphs = {
    'chain3':'0->1->2',
    'fork3':'0->{1-2}',
    'collider3':'{0-1}->2',
    'collider4':'{0-2}->3',
    'collider5':'{0-3}->4',
    'collider6':'{0-4}->5',
    'collider7':'{0-5}->6',
    'collider8':'{0-6}->7',
    'collider9':'{0-7}->8',
    'collider10':'{0-8}->9',
    'collider11':'{0-9}->10',
    'collider12':'{0-10}->11',
    'collider13':'{0-11}->12',
    'collider14':'{0-12}->13',
    'collider15':'{0-13}->14',
    'confounder3':'{0-2}->{0-2}',
    'chain4':'0->1->2->3',
    'chain5':'0->1->2->3->4',
    'chain6':'0->1->2->3->4->5',
    'chain7':'0->1->2->3->4->5->6',
    'chain8':'0->1->2->3->4->5->6->7',
    'chain9':'0->1->2->3->4->5->6->7->8',
    'chain10':'0->1->2->3->4->5->6->7->8->9',
    'chain11':'0->1->2->3->4->5->6->7->8->9->10',
    'chain12':'0->1->2->3->4->5->6->7->8->9->10->11',
    'chain13':'0->1->2->3->4->5->6->7->8->9->10->11->12',
    'chain14':'0->1->2->3->4->5->6->7->8->9->10->11->12->13',
    'chain15':'0->1->2->3->4->5->6->7->8->9->10->11->12->13->14',
    'full3':'{0-2}->{0-2}',
    'full4':'{0-3}->{0-3}',
    'full5':'{0-4}->{0-4}',
    'full6':'{0-5}->{0-5}',
    'full7':'{0-6}->{0-6}',
    'full8':'{0-7}->{0-7}',
    'full9':'{0-8}->{0-8}',
    'full10':'{0-9}->{0-9}',
    'full11':'{0-10}->{0-10}',
    'full12':'{0-11}->{0-11}',
    'full13':'{0-12}->{0-12}',
    'full14':'{0-13}->{0-13}',
    'full15':'{0-14}->{0-14}',
    'tree9':'0->1->3->7,0->2->6,1->4,3->8,2->5',
    'tree10':'0->1->3->7,0->2->6,1->4->9,3->8,2->5',
    'tree11':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5',
    'tree12':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11',
    'tree13':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12',
    'tree14':'0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'tree15':'0->1->3->7,0->2->6->14,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'jungle3':'0->{1-2}',
    'jungle4':'0->1->3,0->2,0->3',
    'jungle5':'0->1->3,1->4,0->2,0->3,0->4',
    'jungle6':'0->1->3,1->4,0->2->5,0->3,0->4,0->5',
    'jungle7':'0->1->3,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6',
    'jungle8':'0->1->3->7,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7',
    'jungle9':'0->1->3->7,3->8,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8',
    'jungle10':'0->1->3->7,3->8,1->4->9,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9',
    'jungle11':'0->1->3->7,3->8,1->4->9,4->10,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10',
    'jungle12':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11',
    'jungle13':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12',
    'jungle14':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13',
    'jungle15':'0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,6->14,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13,2->14',
    'bidiag3':'{0-2}->{0-2}',
    'bidiag4':'{0-1}->{1-2}->{2-3}',
    'bidiag5':'{0-1}->{1-2}->{2-3}->{3-4}',
    'bidiag6':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}',
    'bidiag7':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}',
    'bidiag8':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}',
    'bidiag9':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}',
    'bidiag10':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}',
    'bidiag11':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}',
    'bidiag12':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}',
    'bidiag13':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}',
    'bidiag14':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}',
    'bidiag15':'{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}->{13-14}',
}


def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width // 2], [c0 + width // 2, c0, c0 + width // 2, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width//2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def cross(r0, c0, width, im_size):
    diff1 = width // 3 + 1
    diff2 = 2 * width // 3
    rr = [r0 + diff1, r0 + diff2, r0 + diff2, r0 + width, r0 + width, r0 + diff2, r0 + diff2, r0 + diff1, r0 + diff1, r0, r0, r0 + diff1]
    cc = [c0, c0, c0 + diff1, c0 + diff1, c0 + diff2, c0 + diff2, c0 + width, c0 + width, c0 + diff2, c0 + diff2, c0 + diff1, c0 + diff1]
    return skimage.draw.polygon(rr, cc, im_size)

def pentagon(r0, c0, width, im_size):
    diff1 = width // 3 - 1
    diff2 = 2 * width // 3 + 1
    rr = [r0 + width // 2, r0 + width, r0 + width, r0 + width // 2, r0]
    cc = [c0, c0 + diff1, c0 + diff2, c0 + width, c0 + width // 2]
    return skimage.draw.polygon(rr, cc, im_size)

def parallelogram(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0 + width // 2, c0 + width, c0 + width - width // 2]
    return skimage.draw.polygon(rr, cc, im_size)

def scalene_triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width//2], [c0 + width - width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)

def render_cubes(objects, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)
    cols = ['purple', 'green', 'orange', 'blue', 'brown']
    for i, pos in objects.items():
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS)) # Crop and resize
    return im / 255.


class MLP_Color(nn.Module):
    def __init__(self, dims, force_change=True):
        super().__init__()
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))
            torch.nn.init.orthogonal_(self.layers[-1].weight.data, 3.5)
            torch.nn.init.uniform_(self.layers[-1].bias.data, -2.1, +2.1)
        self.layers = nn.ModuleList(self.layers)
        self.force_change = force_change

    def forward(self, x, mask, current_color=None):
        # mask is used to remove non-parent nodes
        x = x * mask
        for i, l in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.softmax(l(x), dim=1)
            else:
                x = torch.relu(l(x))

        # mask out the current color to make sure the intervened color is different
        if self.force_change and current_color is not None:
            x[0, current_color] = 0.0

        max_idx = torch.argmax(x)
        #max_idx = (current_color+1) % x.shape[1]
        determinstic_x = torch.zeros_like(x)
        determinstic_x[0, max_idx] = 1
        return determinstic_x


@dataclass
class Coord:
    x: int
    y: int

    def __add__(self, other):
        return Coord(self.x + other.x, self.y + other.y)


@dataclass
class Object:
    pos: Coord
    color: int


class ColorChangingRL(gym.Env):
    """Gym environment for block pushing task."""
    def __init__(self, test_mode='IID', width=5, height=5, render_type='cubes', *, num_objects=5, num_colors=None,  movement='Dynamic', max_steps=50, seed=None):
        # np.random.seed(0)
        # torch.manual_seed(0)
        self.width = width    # the width of grid world (for true state)
        self.height = height  # the height of grid world (for true state)
        self.render_type = render_type
        self.num_objects = num_objects
        self.test_mode = test_mode
        assert self.test_mode in ['IID', 'OOD-S'], 'only IID and OOD-S are supportted'

        # dynamic means the positions are different in different episodes
        self.movement = movement
        self.use_render = False
        self.instantaneous_effect = False
        self.force_change = True

        if num_colors is None:
            num_colors = num_objects
        self.num_colors = num_colors
        self.num_actions = self.num_objects * self.num_colors
        self.num_target_interventions = max_steps
        self.max_steps = max_steps

        print('enable instantaneous effect:', self.instantaneous_effect)
        print('force_change:', self.force_change)
        print('num_objects:', self.num_objects)
        print('num_colors:', self.num_colors)
        print('max_steps:', self.max_steps)
        print('test_mode:', self.test_mode)
        
        self.mlps = []
        self.mask = None

        colors = ['blue', 'green', 'yellow', 'white', 'red']
        self.colors, _ = get_colors_and_weights(cmap='Set1', num_colors=self.num_colors) 
        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()
        self.adjacency_matrix = None

        # MLPs are used for randomlized intervention to color
        mlp_dims = [self.num_objects * self.num_colors, 4 * self.num_objects, self.num_colors]
        self.mlps = []
        for i in range(self.num_objects):
            self.mlps.append(MLP_Color(mlp_dims, force_change=self.force_change))

        num_nodes = self.num_objects
        num_edges = np.random.randint(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        self.adjacency_matrix = random_dag(num_nodes, num_edges)
        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()

        # Generate masks so that each variable only recieves input from its parents.
        self.generate_masks()

        # If True, then check for collisions and don't allow two objects to occupy the same position.
        self.collisions = True
        self.actions_to_target = []

        self.objects = OrderedDict()
        # Randomize object position.
        fixed_object_to_position_mapping = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2), (1,1), (1, 3), (3, 1), (3,3), (0, 2)]
        while len(self.objects) < self.num_objects:
            idx = len(self.objects)
            # Re-sample to ensure objects don't fall on same spot.
            while not (idx in self.objects and self.valid_pos(self.objects[idx].pos, idx)):
                self.objects[idx] = Object(
                    pos=Coord(x=fixed_object_to_position_mapping[idx][0], y=fixed_object_to_position_mapping[idx][1]),
                    color=torch.argmax(self.object_to_color[idx]))

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 50, 50), dtype=np.float32)
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # load the graph structure and inverntion MLPs
    def load_save_information(self, save):
        self.adjacency_matrix = save['graph']
        for i in range(self.num_objects):
            self.mlps[i].load_state_dict(save['mlp' + str(i)])
        self.generate_masks()
        self.reset()

    def set_graph(self, g):
        if g in graphs.keys():
            print('INFO: Loading predefined graph for configuration '+str(g))
            g = graphs[g]
        num_nodes = self.num_objects
        num_edges = np.random.randint(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        self.adjacency_matrix = random_dag(num_nodes, num_edges, g=g)
        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()
        print(self.adjacency_matrix)
        self.generate_masks()
        self.reset()

    def get_save_information(self):
        save = {}
        save['graph'] = self.adjacency_matrix
        for i in range(self.num_objects):
            save['mlp' + str(i)] = self.mlps[i].state_dict()
        return save

    def render_grid(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[obj.color][:3]
        return im

    def render_circles(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.circle(
                obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = self.colors[obj.color][:3]
        return im.transpose([2, 0, 1])

    def render_shapes(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if idx == 0:
                rr, cc = skimage.draw.circle(obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 1:
                rr, cc = triangle(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 2:
                rr, cc = square(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 3:
                rr, cc = diamond(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 4:
                rr, cc = pentagon(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 5:
                rr, cc = cross(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 6:
                rr, cc = parallelogram(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 7:
                rr, cc = scalene_triangle(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 8:
                rr, cc = square(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 9:
                rr, cc = diamond(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]

        return im.transpose([2, 0, 1])

    def render_grid_target(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
        return im

    def render_circles_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.circle(obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
            im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
        return im.transpose([2, 0, 1])

    def render_shapes_target(self):
        im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
        for idx, obj in self.objects.items():
            if idx == 0:
                rr, cc = skimage.draw.circle(obj.pos.x * 10 + 5, obj.pos.y * 10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 1:
                rr, cc = triangle(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 2:
                rr, cc = square(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 3:
                rr, cc = diamond(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 4:
                rr, cc = pentagon(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 5:
                rr, cc = cross(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 6:
                rr, cc = parallelogram(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
            elif idx == 7:
                rr, cc = scalene_triangle(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 8:
                rr, cc = square(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]
            elif idx == 9:
                rr, cc = diamond(obj.pos.x * 10, obj.pos.y * 10, 10, im.shape)
                im[rr, cc, :] = self.colors[obj.color][:3]

        return im.transpose([2, 0, 1])

    def render_cubes(self):
        im = render_cubes(self.objects, self.width)
        return im.transpose([2, 0, 1])

    def render(self):
        return np.concatenate((
            dict(
                grid=self.render_grid,
                circles=self.render_circles, 
                shapes=self.render_shapes, 
                cubes=self.render_cubes)[self.render_type](), 
            dict(
                grid=self.render_grid_target, 
                circles=self.render_circles_target, 
                shapes=self.render_shapes_target, 
                cubes=self.render_cubes,)[self.render_type]()
        ), axis=0) 

    # get the true state
    def get_state(self):
        # use 2D grid world to represent the true state
        im = np.zeros((self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        im_target = np.zeros((self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        for idx, obj in self.objects.items():
            im[idx * self.num_colors + obj.color, obj.pos.x, obj.pos.y] = 1
            im_target[idx * self.num_colors + torch.argmax(self.object_to_color_target[idx]).item(), obj.pos.x, obj.pos.y] = 1
        return im, im_target

    def generate_masks(self):
        mask = self.adjacency_matrix.unsqueeze(-1)
        mask = mask.repeat(1, 1, self.num_colors)
        self.mask = mask.view(self.adjacency_matrix.size(0), -1)
    
    def generate_target(self, num_steps=10):
        self.actions_to_target = []
        for i in range(num_steps):
            # randomly select the intervened object and intervened color
            intervention_id = random.randint(0, self.num_objects - 1)
            to_color = random.randint(0, self.num_colors - 1)
            self.actions_to_target.append(intervention_id * self.num_colors + to_color)
            self.object_to_color_target[intervention_id] = torch.zeros(self.num_colors)
            self.object_to_color_target[intervention_id][to_color] = 1
            self.sample_variables_target(intervention_id)

    def sample_variables(self, idx, do_everything=False):
        """
        idx: variable at which intervention is performed
        """
        reached = [idx]
        for v in range(idx + 1, self.num_objects):
            if do_everything or self.is_reachable(v, reached):
                if self.instantaneous_effect:
                    reached.append(v)
                if self.test_mode == 'OOD-S' and self.stage == 'test':
                    inp = self.mask[v].cpu().numpy() * np.concatenate(self.object_to_color)
                    new_color_idx = 0
                    for i in range(inp.shape[0]):
                        new_color_idx += (i+1) * (inp[i] + 2)
                    new_color_idx = int(new_color_idx) % self.num_colors
                    self.object_to_color[v] = torch.zeros_like(self.object_to_color[v])
                    self.object_to_color[v][new_color_idx] = 1
                else:
                    inp = torch.cat(self.object_to_color, dim=0).unsqueeze(0)
                    mask = self.mask[v].unsqueeze(0)
                    current_color = torch.argmax(self.object_to_color[v])
                    out = self.mlps[v](inp, mask, current_color)
                    self.object_to_color[v] = out.squeeze(0)

    def sample_variables_target(self, idx, do_everything=False):
        """
        idx: variable at which intervention is performed
        """
        reached = [idx]
        # start from next object, current color will not be changed
        for v in range(idx + 1, self.num_objects):
            if do_everything or self.is_reachable(v, reached):
                if self.instantaneous_effect:
                    reached.append(v)
                # use current color as input to get the target color
                inp = torch.cat(self.object_to_color_target, dim=0).unsqueeze(0)
                mask = self.mask[v].unsqueeze(0)
                current_color = torch.argmax(self.object_to_color_target[v])
                out = self.mlps[v](inp, mask, current_color)
                self.object_to_color_target[v] = None
                self.object_to_color_target[v] = out.squeeze(0)

    def check_softmax(self):
        s_ = []
        for i in range(1, len(self.objects)):
            x = torch.cat(self.object_to_color, dim=0).unsqueeze(0)
            mask = self.mask[i].unsqueeze(0)
            _, s = self.mlps[i](x, mask, return_softmax=True)
            s_.append(s.detach().cpu().numpy().tolist())
        return s_
        
    def check_softmax_target(self):
        s_ = []
        for i in range(1, len(self.objects)):
            x = torch.cat(self.object_to_color_target, dim=0).unsqueeze(0)
            mask = self.mask[i].unsqueeze(0)
            _, s = self.mlps[i](x, mask, return_softmax=True)
            s_.append(s.detach().cpu().numpy().tolist())
        return s_

    def reset(self, num_steps=10, graph=None, stage='train'):
        self.cur_step = 0
        self.stage = stage

        # reset interventon color and target color
        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]
        self.object_to_color_target = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]

        # Sample color for root node randomly
        root_color = np.random.randint(0, self.num_colors)
        self.object_to_color[0][root_color] = 1
        # Sample color for other nodes using MLPs
        self.sample_variables(0, do_everything=True)

        if self.movement == 'Dynamic':
            self.objects = OrderedDict()
            # Randomize object position.
            while len(self.objects) < self.num_objects:
                idx = len(self.objects)
                # Re-sample to ensure objects don't fall on same spot.
                while not (idx in self.objects and self.valid_pos(self.objects[idx].pos, idx)):
                    self.objects[idx] = Object(
                        pos=Coord(x=np.random.choice(np.arange(self.width)), y=np.random.choice(np.arange(self.height))),
                        color=torch.argmax(self.object_to_color[idx]))
        for idx, obj in self.objects.items():
            obj.color = torch.argmax(self.object_to_color[idx])
        #self.sample_variables_target(0, do_everything = True)
        
        if self.test_mode == 'OOD-S' and self.stage == 'test':
            # set all target color to the same one
            target_color = np.random.randint(0, self.num_colors)
            for i in range(len(self.object_to_color_target)):
                self.object_to_color_target[i][target_color] = 1
        else:
            # initial the target color with current color
            for i in range(len(self.object_to_color)):
                self.object_to_color_target[i][torch.argmax(self.object_to_color[i])] = 1
            self.generate_target(num_steps)
            #self.check_softmax()
            #self.check_softmax_target()

        # get state
        state_in, state_target = self.get_state()

        # get observation
        if self.use_render:
            observations = self.render()
            observation_in, observations_target = observations[:3, :, :], observations[3:, :, :]
            return (state_in, observation_in), (state_target, observations_target)
        else:
            # we return the target in step() instead of reset()
            state_all = self.augment_state_with_goal(state_in, state_target)
            return state_all

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos.x not in range(0, self.width):
            return False
        if pos.y not in range(0, self.height):
            return False

        if self.collisions:
            for idx, obj in self.objects.items():
                if idx == obj_id:
                    continue

                if pos == obj.pos:
                    return False

        return True

    def is_reachable(self, idx, reached):
        for r in reached:
            if self.adjacency_matrix[idx, r] == 1:
                return True
        return False

    def translate(self, obj_id, color_id):
        """ Get the color of obejct

        Args:
            obj_id: ID of object.
            color_id: ID of color.
        """
        color_ = torch.zeros(self.num_colors)
        color_[color_id] = 1
        self.object_to_color[obj_id] = color_
        self.sample_variables(obj_id)
        for idx, obj in self.objects.items():
            obj.color = torch.argmax(self.object_to_color[idx])

    def random_action(self):
        action_idx = np.random.randint(0, self.action_space.n)
        action = np.zeros((self.action_space.n,))
        action[action_idx] = 1.0
        return action

    def step(self, action):
        # convert onehot (action_dim,) to action
        action = np.argmax(action)

        # only change color of one object one time
        obj_id = action // self.num_colors
        color_id = action % self.num_colors 

        # check episode finish
        done = False
        if self.cur_step >= self.max_steps:
            done = True
        self.cur_step += 1
        
        # change the color of one object
        self.translate(obj_id, color_id)

        # dense reward, check if all object color matches
        matches = 0
        for c1, c2 in zip(self.object_to_color, self.object_to_color_target):
            if torch.argmax(c1).item() == torch.argmax(c2).item():
                matches += 1
        #reward = matches / self.num_objects

        # goal-conditioned sparse reward
        reward = 0
        if matches == self.num_objects:
            reward = 1
            done = True # early stop if we achieve the goal

        # get state
        state, target = self.get_state()  # input state dimension [object_num * color_num, width, height]

        # get observation
        if self.use_render:
            state_obs = self.render()
            state_obs = state_obs[:3, :, :] # input image dimension [3, ]
            return (state, state_obs), reward, done, None
        else:
            # [object_num * color_num, width, height] -> [object_num * color_num * width * height * 2]
            state = self.augment_state_with_goal(state, target)
            return state, reward, done, None

    def augment_state_with_goal(self, state, target):
        state = state.reshape(self.num_objects * self.num_colors * self.width * self.height,)
        target = target.reshape(self.num_objects * self.num_colors * self.width * self.height,)
        augmented_state = np.concatenate([state, target], axis=0)
        return augmented_state

    def sample_step(self, action: int):
        obj_id = action // self.num_colors
        color_id = action % self.num_colors 

        done = False
        objects = self.objects.copy()
        object_to_color = self.object_to_color.copy()
        self.translate(obj_id, color_id)
        matches = 0
        for c1, c2 in zip(self.object_to_color, self.object_to_color_target):
            if torch.argmax(c1).item() == torch.argmax(c2).item():
                matches += 1
        reward = 0
        self.objects = objects
        self.object_to_color = object_to_color

        state_obs = self.render()
        state_obs = state_obs[:3, :, :]
        if self.cur_step >= self.max_steps:
            done = True
        reward = matches / self.num_objects
        self.cur_step += 1
        return reward, state_obs

    def sample_step_1(self):
        # access the true action trajectory towards the target
        action = self.actions_to_target[0]
        self.actions_to_target = self.actions_to_target[1:]
        return action


# ---------------------------------------------------------------------------------
# Main Training Script
# ---------------------------------------------------------------------------------

def train_agent(exp_name, mode='IID', agent_type='GRADER', grader_model='causal', env_name='chemistry', 
                graph='chain', num_episodes=200, test_episode=100, test_interval=10, save_interval=10000,
                config_path="config/chemistry_config.yaml"):  
    """
    主训练函数，训练和测试智能体
    
    参数:
        exp_name: 实验名称
        mode: 'IID' 或 'OOD-S'，表示数据分布
        agent_type: 'GRADER' 或 'SAC'，智能体类型
        grader_model: 'causal', 'full', 'mlp', 或 'gnn'，GRADER模型类型
        env_name: 环境名称，当前仅支持'chemistry'
        graph: 图类型，'chain', 'full', 'jungle', 'collider'等
        num_episodes: 训练的总回合数
        test_episode: 每次测试的回合数
        test_interval: 每多少个训练回合进行一次测试
        save_interval: 每多少个训练回合保存一次模型
    """
    # 环境参数设置
    if env_name == 'chemistry':
        num_steps = 10
        movement = 'Static'  # Dynamic 或 Static
        if mode == 'IID':
            num_objects = 5
            num_colors = 5
        else:
            num_objects = 5
            num_colors = 5
        width = 5
        height = 5
        graph_name = graph + str(num_objects)  # chain, full
        env = ColorChangingRL(
            test_mode=mode, 
            render_type='shapes', 
            num_objects=num_objects, 
            num_colors=num_colors, 
            movement=movement, 
            max_steps=num_steps
        )
        env.set_graph(graph_name)

        # 保存图结构和MLP参数
        env_data = env.get_save_information()
        env.load_save_information(env_data)

    
        try:
            config = load_config(config_path=config_path)
            agent_config = config[agent_type]
        except Exception as e:
            print(f"Warning: {e}")
            # 直接使用默认配置，无需再次调用load_config
            config = load_config(config_path="")  # 传递空字符串以确保返回默认配置
        agent_config = config[agent_type]



        env_params = {
            'action_dim': env.action_space.n,
            'num_colors': env.num_colors,
            'num_objects': env.num_objects,
            'width': env.width,
            'height': env.height,
            'state_dim': env.num_colors * env.num_objects * env.width * env.height * 2,
            'goal_dim': env.num_colors * env.num_objects * env.width * env.height,
            'adjacency_matrix': env.adjacency_matrix,  # 存储图结构
        }
    else:
        raise ValueError('Wrong environment name')
    
    env_params['env_name'] = env_name
    agent_config['env_params'] = env_params
    
    # 创建保存路径
    save_path = os.path.join('./log', exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = agent_config['model_path']
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    render = False  # 是否渲染环境
    trails = 1      # 运行的实验次数

    for t_i in range(trails):
        # 创建智能体
        if agent_type == 'GRADER':
            agent_config['grader_model'] = grader_model
            agent = GRADER(agent_config)
        elif agent_type == 'SAC':
            agent = SAC(agent_config)

        save_gif_count = 0
        test_reward = []
        train_reward = []
        
        # 训练循环
        for e_i in range(num_episodes):
            state = env.reset(stage='train')
            done = False
            one_train_reward = 0
            
            # 单个回合的交互循环
            while not done:
                action = agent.select_action(env, state, False)
                next_state, reward, done, info = env.step(action)
                one_train_reward += reward

                if agent.name in ['SAC']: 
                    agent.store_transition([state, action, reward, next_state, done])
                    agent.train()
                elif agent.name in ['GRADER']:
                    agent.store_transition([state, action, next_state])

                state = copy.deepcopy(next_state)

            # 回合结束后训练GRADER
            if agent.name in ['GRADER']: 
                agent.train()
            train_reward.append(one_train_reward)

            # 保存模型
            if (e_i+1) % save_interval == 0:
                agent.model_id = e_i + 1
                if agent.name == 'GRADER':
                    agent.save_model()
                else:
                    agent.save_model()

            # 测试阶段
            if (e_i+1) % test_interval == 0:
                test_reward_mean = []
                for t_j in range(test_episode):
                    state = env.reset(stage='test')
                    done = False
                    total_reward = 0
                    step_reward = []
                    
                    # 单个测试回合
                    while not done:
                        action = agent.select_action(env, state, True)
                        next_state, reward, done, info = env.step(action)

                        if render:
                            env.render()
                            time.sleep(0.05)

                        state = copy.deepcopy(next_state)
                        total_reward += reward
                        step_reward.append(reward)
                    
                    test_reward_mean.append(total_reward)

                test_reward_mean = np.mean(test_reward_mean, axis=0)
                print('[{}/{}] [{}/{}] Test Reward: {}'.format(t_i, trails, e_i, num_episodes, test_reward_mean))
                test_reward.append(test_reward_mean)
                np.save(os.path.join(save_path, f'test.reward.{t_i}.npy'), test_reward)
    
    return test_reward


# 如果直接运行此脚本
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='grader_test', help='Experiment name')
    parser.add_argument('--mode', type=str, default='IID', choices=['IID', 'OOD-S'], help='IID means i.i.d. samples and OOD-S means spurious correlation')
    parser.add_argument('--agent', type=str, default='GRADER', choices=['GRADER', 'SAC'], help='Agent type')
    parser.add_argument('--grader_model', type=str, default='causal', choices=['causal', 'full', 'mlp', 'gnn'], help='Type of model used in GRADER')
    parser.add_argument('--env', type=str, default='chemistry', help='Name of environment')
    parser.add_argument('--graph', type=str, default='chain', choices=['collider', 'chain', 'full', 'jungle'], help='Type of groundtruth graph in chemistry')
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--test_interval', type=int, default=10, help='Test interval')
    
    args = parser.parse_args()
    
    # 运行训练
    train_agent(
        exp_name=args.exp_name,
        mode=args.mode,
        agent_type=args.agent,
        grader_model=args.grader_model,
        env_name=args.env,
        graph=args.graph,
        num_episodes=args.episodes,
        test_episode=args.test_episodes,
        test_interval=args.test_interval
    )