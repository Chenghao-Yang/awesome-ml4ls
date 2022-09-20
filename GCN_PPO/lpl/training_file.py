import os
# import gym
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from copy import deepcopy
# from .scl_session import SCLSession as SCLGame
from .environment import LogicSession as SCLGame
import yaml
import argparse
from dgl.nn.pytorch import GraphConv
import dgl
import csv
import torch.nn.functional as F
from torchsummary import summary

# gym.logger.set_level(40)  # Block warning

"""net.py"""
# device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

"""
designSet1 = ['/home/yangch/EPFL/arithmetic/adder.aig', '/home/yangch/EPFL/arithmetic/bar.aig', '/home/yangch/EPFL/arithmetic/log2.aig', '/home/yangch/EPFL/arithmetic/max.aig', '/home/yangch/EPFL/arithmetic/multiplier.aig', '/home/yangch/EPFL/arithmetic/sin.aig', '/home/yangch/EPFL/arithmetic/sqrt.aig','/home/yangch/EPFL/arithmetic/square.aig']
designSet2 = ['/home/yangch/EPFL/random_control/cavlc.aig','/home/yangch/EPFL/random_control/ctrl.aig','/home/yangch/EPFL/random_control/i2c.aig','/home/yangch/EPFL/random_control/int2float.aig', '/home/yangch/EPFL/random_control/mem_ctrl.aig', '/home/yangch/EPFL/random_control/priority.aig', '/home/yangch/EPFL/random_control/router.aig', '/home/yangch/EPFL/random_control/voter.aig']
"""
test_designs = ['/home/yangch/EPFL/random_control/i2c.aig', '/home/yangch/EPFL/random_control/cavlc.aig',
                '/home/yangch/EPFL/random_control/ctrl.aig', '/home/yangch/EPFL/random_control/int2float.aig',
                '/home/yangch/EPFL/random_control/priority.aig', '/home/yangch/EPFL/random_control/router.aig']

design_one = ['/home/yangch/EPFL/random_control/ctrl.aig']

designs = design_one


def log(message):
    print('[LSJedi {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)


class Normalizer():
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = torch.clone(self.mean)
        x = torch.from_numpy(x)
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2, max=1000000000)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        # self.mean = torch.from_numpy(self.mean)
        inputs = torch.from_numpy(inputs)
        normalize_val = (inputs - self.mean) / obs_std
        return normalize_val

    def reset(self):
        self.n = torch.zeros(self.num_inputs)
        self.mean = torch.zeros(self.num_inputs)
        self.mean_diff = torch.zeros(self.num_inputs)
        self.var = torch.zeros(self.num_inputs)


def gcn_message(edges):
    return {
        'msg': edges.src['h']
    }


def gcn_reduce(nodes):
    # Theoritically, the sum operation is better than the average
    # Or use attention (GAT)
    return {
        'h': torch.sum(nodes.mailbox['msg'], dim=1)
    }


class GraphConvolutionLayer(nn.Module):
    # The depth of this layer is irrelevant of the depth in the GCN module.
    # This is the depth of the NN that does the aggregation.
    # Below in GCN module is the depth of the GCN, which goes deep in the graph itself.
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.update_all(gcn_message, gcn_reduce)
        h = g.ndata.pop('h')

        return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_size, embedding_size):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolutionLayer(in_features, hidden_size)
        self.gcn2 = GraphConvolutionLayer(hidden_size, hidden_size)
        self.gcn3 = GraphConvolutionLayer(hidden_size, embedding_size)

    def forward(self, g, inputs):
        h = F.relu(self.gcn1(g, inputs))
        h = F.relu(self.gcn2(g, h))
        h = self.gcn3(g, h)

        # report graph state vector
        g.ndata['h'] = F.relu(h)
        graph_embedding = dgl.mean_nodes(g, 'h')
        g.ndata.pop('h')

        return graph_embedding


"""
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.conv3 = GraphConv(hidden_size, hidden_size)
        self.conv4 = GraphConv(hidden_size, out_len)

    def forward(self, g):
        h = self.conv1(g, g.ndata['inv'])
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv4(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return torch.squeeze(hg)
"""

"""
class FcGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 28)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)
        self.gcn = GCN(6, 12, 4)

    def forward(self, x, graph):
        graph_state = self.gcn(graph)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0))
        x = self.act2(x)
        x = self.fc3(x)
        return x
"""


class ShareBody(nn.Module):
    def __init__(self, state_dim, outFeats, lstm_hidden_dim):
        super(ShareBody, self).__init__()
        self._numFeatures = state_dim
        self._outFeats = outFeats
        self.fc1 = nn.Linear(state_dim, 28)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outFeats)
        self.gcn = GCN(2, 12, 4)
        self.lstm = nn.LSTM(32, lstm_hidden_dim, batch_first=True, bidirectional=False)
        self.w_omega = nn.Parameter(torch.Tensor(
            lstm_hidden_dim * 2, lstm_hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(lstm_hidden_dim * 2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x, graph):

        if type(graph) == list:
            i = 0
            for n in graph:
                i = i + 1
                # print(i)
                h = torch.cat((n.in_degrees().view(-1, 1).float(), n.out_degrees().view(-1, 1).float()), 1).to(device)
                graph_state_fornow = self.gcn(n, h)
                if i < 2:
                    graph_state = graph_state_fornow
                else:
                    graph_state = torch.cat((graph_state, graph_state_fornow), dim=0)
        else:
            h = torch.cat((graph.in_degrees().view(-1, 1).float(), graph.out_degrees().view(-1, 1).float()), 1).to(
                device)
            graph_state = self.gcn(graph, h)
        """
        h = torch.cat((graph.in_degrees().view(-1, 1).float(), graph.out_degrees().view(-1, 1).float()), 1).to(
            device)
        graph_state = self.gcn(graph, h)
        """
        # print(graph_state)
        x = self.fc1(x)
        x = self.act1(x)
        # print(graph_state)
        # x = x.unsqueeze(0)
        # print('x.size---->', x.size())
        # print('graph_state---->', graph_state.size())
        x = torch.cat((x, graph_state), 1)
        # x = self.fc2(x)
        # x = self.act2(x)
        # size = x.shape
        #"""
        x = x.view(1, x.size()[0], x.size()[1])
        #h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        #c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(x)
        #output = torch.chunk(output, 5, dim=1)[-1]
        # Attention过程
        x = output
        x = x.squeeze(0)

        print('lstm_output---->', x.size())
        u = torch.tanh(torch.matmul(x, self.w_omega))
        print('u---->', u)
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        print('att---->', att,att.size())
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        print('att_score---->', att_score)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        #print('scored_x--->', scored_x)
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        feat = torch.sum(scored_x, dim=1)  # 加权求和
        #print('attention_size---->', feat.size())
        #output = torch.squeeze(output, 1)
        #hn = hn.squeeze(0)
        #joint_state = torch.cat([self_state, hn], dim=1)

        # x = self.fc3(x)
        #"""
        return x


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, mid_dim), nn.ReLU(),
                                 # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical
        self.body = ShareBody(state_dim, 32, 32)

    def forward(self, state, graph):
        # print(type(graph))
        state = state.to(torch.float32)
        x = self.body(state, graph)
        x = self.net(x)
        return x  # action_prob without softmax

    def get_action(self, state_statistics, graph):
        out = self.forward(state_statistics, graph)
        a_prob = self.soft_max(out)
        print('a_prob--->', a_prob)
        # action = Categorical(a_prob).sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(1)
        print(action)
        return action, a_prob

    def get_logprob_entropy(self, state_statistics, graph, a_int):
        out = self.forward(state_statistics, graph)
        # state = state.to(torch.float32)
        a_prob = self.soft_max(out)

        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, mid_dim), nn.ReLU(),
                                 # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))
        self.body = ShareBody(state_dim, 32, 32)
        self.state_dim = state_dim

    def forward(self, state, graph):
        if type(state) != torch.Tensor:
            state = np.reshape(state, (1, self.state_dim))
            state = torch.as_tensor(state, dtype=torch.float32)
        else:
            # print(state)
            pass
        # state = state.to(torch.float32)
        x = self.body(state, graph)
        x = self.net(x)
        return x  # Advantage value


"""agent.py"""


class AgentPPO:
    def __init__(self):
        super().__init__()
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = True
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

        '''init modify'''
        self.ClassCri = CriticAdv
        self.ClassAct = ActorDiscretePPO

        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.01~0.05
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.trajectory_list = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0, env_num=1):
        # self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.trajectory_list = [list() for _ in range(env_num)]
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state_statistics, graph):
        # print(state)
        # state = np.array([1,2,3,4,5,6,7,8,9])
        # print(state)
        # state_statistics = state_statistics.numpy()
        # print(graph)
        states_statistics = torch.as_tensor((state_statistics,), dtype=torch.float32, device=self.device)
        #print('states_statistics--->', states_statistics, states_statistics.size())
        actions, noises = self.act.get_action(states_statistics, graph)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    """
    def explore_env(self, env, target_step):
        trajectory_temp = list()

        state = self.state
        last_done = 0
        for i in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))
            trajectory_temp.append((state, reward, done, action, noise))
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        trajectory_list = self.trajectory_list[0] + trajectory_temp[:last_done + 1]
        self.trajectory_list[0] = trajectory_temp[last_done:]
        return trajectory_list
    """

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau, options):
        with torch.no_grad():
            # state_len = buffer[0][0]
            # state_len = np.array(state_len)
            # print(self.act.net[0].weight)

            buf_len = options['iterations']

            # buf_len = buffer[0][0].shape[0]
            buf_state = buffer[0]
            # print('buf_state', buf_state)

            # print(buf_state[0][0])
            # print(buf_state[0][1])
            buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer[1:]]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer
            # print(buf_action)

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 0  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i][0], buf_state[i][1]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_state_statistics = [buf_state[i][0] for i in range(0, buf_len)]
            buf_state_statistics = torch.as_tensor(buf_state_statistics, dtype=torch.float32)
            buf_graph = [buf_state[i][1] for i in range(0, buf_len)]

            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:]

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None

        for _ in range(5):  # (int(buf_len / batch_size * repeat_times)):
            #indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)
            #indices = torch.randperm(buf_len)
            #print(indices)
            # print('indices--->', indices)
            """
            state = buf_state_statistics[indices]
            # state = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            #state = state.unsqueeze(0)
            #buf_graph = [buf_state[i][1] for i in range(0, buf_len)]
            state_graph = [buf_graph[m] for m in indices]
            #state_graph = buf_graph[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]
            """
            state = buf_state_statistics
            state_graph = buf_graph
            #state_graph = buf_graph[indices]
            action = buf_action
            r_sum = buf_r_sum
            logprob = buf_logprob
            advantage = buf_advantage

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, state_graph, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            self.optim_update(self.act_optim, obj_actor)
            print('obj_actor---->', obj_actor)

            value = self.cri(state, state_graph).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            # print('value---->', value)
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None
            #print("到了更新网络？")

            # summary(self.act, input_size=(1, 32, 32), graph = (1,32,32), batch_size=-1)
            # print(self.act.net[0].weight)
            # print('self.cri---->', self.cri)
            # print('self.act---->', self.act)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1))
        # print('####################')
        # print(self.act.net[0].weight)
        return obj_critic.item(), obj_actor.item(), a_std_log.mean().item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])  # fix a bug here
            pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1.0 - tau))


class Trajectory(object):
    """
    @brief The experience of a trajectory
    """

    def __init__(self, states, rewards, dones, actions, a_probs):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.dones = dones
        self.actions = actions
        self.a_probs = a_probs

    def __lt__(self, other):
        return self.value < other.value


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.ClassAct = ActorDiscretePPO
        # self.ClassCri = CriticAdv

    def explore_env(self, env, target_step):
        trajectory_temp = list()
        states, rewards, actions, dones, a_probs = [], [0], [], [], []
        state = self.state
        # print("未标准化化前的状态")
        # print(state)
        # normalize = Normalizer(env.state_dim)
        # normalize.reset()
        # normalize.observe(x=state[0])
        # nor_state_statistics = normalize.normalize(inputs=state[0])
        # print("标准化后的状态")
        # print(state)

        last_done = 0
        for i in range(target_step):
            log('Iteration: ' + str(i + 1))
            done = 0
            action, a_prob = self.select_action(state[0], state[1])  # different
            a_int = int(action)  # different
            next_state, reward, done, _ = env.step(a_int)  # different
            trajectory_temp.append((state, reward, done, a_int, a_prob))  # different

            states.append(state)
            rewards.append(reward)
            actions.append(a_int)
            dones.append(done)
            a_probs.append(a_prob)
            Trajectory_to_update = Trajectory(states, rewards, dones, actions, a_probs)

            state = next_state
            # normalize.observe(x=state[0])
            # nor_state_statistics = normalize.normalize(inputs=state[0])

            """
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
            """
        # self.state = state

        '''splice list'''
        # print('附加信息', trajectory_temp)
        # trajectory_list = self.trajectory_list[0] + trajectory_temp[:last_done + 1]
        trajectory_list = trajectory_temp
        # print('trajectory list', trajectory_list)
        # print('self.trajectory_list', self.trajectory_list)
        self.trajectory_list[0] = trajectory_temp[last_done:]
        # print('self.trajectory_list[0]', self.trajectory_list[0])
        # print('#############')
        # print("return")
        # return trajectory_list
        return Trajectory_to_update


'''run.py'''


class Arguments:
    def __init__(self, agent=None, env=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        # self.env = env  # the environment for training
        self.env = env

        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.target_episode = 50

        if if_on_policy:  # (on-policy)
            self.net_dim = 32  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 5  # 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 4  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 5  # 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2  # number of times that get episode return in first
        self.eval_times2 = 4  # number of times that get episode return in second
        self.random_seed = 1  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main):
        """
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}_{self.env.env_name}_{self.visible_gpu}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)
        """

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)


def train_and_evaluate(args, options, agent_id=0):
    args.init_before_training(if_main=True)

    act_save_path = options['actor_model_dir']
    cri_save_path = options['critic_model_dir']
    agent_save_path = options['model_dir']
    """
    if not os.path.exists(agent_save_path):
        os.makedirs(agent_save_path)
    """

    '''init: Agent'''
    # env = SCLGame
    # env = args.env
    # env = env
    agent = args.agent
    state_dim = 10
    action_dim = len(options['optimizations'])
    agent.init(args.net_dim, state_dim, action_dim,
               args.learning_rate, args.if_per_or_gae)

    '''init Evaluator'''
    """
    eval_env = deepcopy(env)
    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                          args.eval_times1, args.eval_times2, args.eval_gap)
    """

    '''init ReplayBuffer'''
    buffer = list()

    def update_buffer(_trajectory):
        # _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
        # print(_trajectory)
        # print(type(_trajectory[0]))
        # _trajectory[0] = np.array(_trajectory[0])
        # ten_state = torch.as_tensor(_trajectory[0])
        ten_state = _trajectory.states
        # ten_state = torch.as_tensor(_trajectory.states, dtype=torch.float32)
        # ten_state = torch.as_tensor([item.detach().numpy() for item in _trajectory[0]])
        ten_reward = torch.as_tensor(_trajectory.rewards, dtype=torch.float32) * reward_scale
        ten_mask = (1.0 - torch.as_tensor(_trajectory.dones, dtype=torch.float32)) * gamma  # _trajectory[2] = done
        # print("update_buffer")
        ten_action = torch.as_tensor(_trajectory.actions)
        ten_noise = torch.as_tensor(_trajectory.a_probs, dtype=torch.float32)

        buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)
        # print('buffer', buffer)

        _steps = ten_reward.shape[0]
        _r_exp = ten_reward.mean()
        return _steps, _r_exp

    '''start training'''
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    target_episode = args.target_episode
    reward_scale = args.reward_scale
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    # print("真的有reset吗")
    # agent.state = env.reset()

    dir = options['playground_dir']

    episode_reward = open(os.path.join(dir, "reward" + '.csv'), 'a', newline='')
    # csvFile2 = open('bar.csv', 'a', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(episode_reward)
    writer.writerow(['episode', 'reward'])

    if_train = True
    while if_train:
        for i in range(target_episode):
            log('Episode: ' + str(i + 1))

            file = np.random.choice(designs)
            print('file---->', file)
            env = SCLGame(options, file)

            agent.state = env.reset()

            with torch.no_grad():
                trajectory_list = agent.explore_env(env, target_step)
                # print(trajectory_list)
                steps, r_exp = update_buffer(trajectory_list)

            episode_total_reward = sum(trajectory_list.rewards)
            episode_reward = open(os.path.join(dir, "reward" + '.csv'), 'a', newline='')  # 设置newline，否则两行之间会空一行
            writer = csv.writer(episode_reward)
            writer.writerow([i + 1, episode_total_reward])
            # writer.writerow([i + 1, r_exp.item()])
            episode_reward.close()

            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau, options)

            torch.save(agent, agent_save_path)
            # torch.save(agent.act, act_save_path)
            # torch.save(agent.cri.state_dict(), cri_save_path)
        # for i in 1:
        # logging_tuple
        # if i + 1 == target_episode:
        if_train = False

    episode_reward.close()
    """
        with torch.no_grad():
            # print("进来了？")
            if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
    """

    # print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


class Evaluator:
    def __init__(self, cwd, agent_id, device, env, eval_times1, eval_times2, eval_gap, ):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'
        self.r_max = -np.inf
        self.total_step = 0

        self.env = env
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = 0
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            return False  # if_reach_goal

        self.eval_time = time.time()
        rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                              range(self.eval_times1)]
        r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

        if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
            rewards_steps_list += [get_episode_return_and_step(self.env, act, self.device)
                                   for _ in range(self.eval_times2 - self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                  f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                  f"{'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{self.used_time:>8}  ########")

        print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
              f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
        return if_reach_goal

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = True  # env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        print(episode_step)
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


"""
class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        state, reward, done, info_dict = self.env.step(action * self.action_max)
        return state.astype(np.float32), reward, done, info_dict
"""


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    assert isinstance(env, gym.Env)

    env_name = getattr(env, 'env_name', None)
    env_name = env.unwrapped.spec.id if env_name is None else None

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env, 'target_return', None)
    target_return_default = getattr(env.spec, 'reward_threshold', None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return


'''demo.py'''
"""
parser = argparse.ArgumentParser(description='Performs logic synthesis optimization using RL')

parser.add_argument("params", type=open, nargs='?', default='params.yml', \
                    help="Path to the params.yml file")

args = parser.parse_args()
options = yaml.load(args.params, Loader=yaml.FullLoader)
out_env = SCLGame(options, '/home/yangch/EPFL-benchmarks/arithmetic/adder.aig')
"""


def logic_jedi_master(options):
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()
    args.visible_gpu = '0'


    # args.env = SCLGame(options, '/home/yangch/EPFL-benchmarks/arithmetic/adder.aig')
    args.reward_scale = 2 ** -1
    args.target_step = options['iterations']  # args.env.max_step * 8
    args.target_episode = options['episodes']


    # env = args.env
    # env.reset()

    train_and_evaluate(args, options)





