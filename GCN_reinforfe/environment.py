#!/usr/bin/python3
# Copyright (c) 2022, @Yang Chenghao
# All rights reserved.

import os
import re
import datetime
import math
import numpy as np
import torch
from subprocess import check_output
from features import extract_features
from aig import read_verilog


def log(message):
    print('[LSJedi {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)


class LogicSession:
    """
    A class to represent a logic synthesis optimization session using ABC
    """

    def __init__(self, params, file_name):

        self.file = file_name

        self.target_return = 5

        self.params = params
        self.env_name = "LS"
        self.action_dim = len(self.params['optimizations'])
        self.total_len_seq = self.params['iterations']
        self.state_dim = 10

        # self.action_space_length = len(self.params['optimizations'])
        # self.observation_space_size = 9     # number of features

        abc_command = 'read ' + self.file + '; '
        abc_command += 'strash' + ';'
        abc_command += 'print_stats;'
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        self.initial_node, self.initial_level = self._get_metrics(proc)

        abc_command = 'read ' + self.file + '; '
        abc_command += 'strash' + ';' + 'resyn2' + ';'
        abc_command += 'print_stats;'
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        self.resyn2_node, self.resyn2_level = self._get_metrics(proc)

        self.reward_baseline = (float(self.initial_node) / float(self.initial_node) - float(self.resyn2_node) / float(self.initial_node)) / float(self.total_len_seq)

        self.iteration = 0
        self.episode = 0
        self.sequence = ['strash']
        self.node, self.levels = float('inf'), float('inf')

        self.best_known_node = (float('inf'), float('inf'), -1, -1)
        self.best_known_levels = (float('inf'), float('inf'), -1, -1)
        self.best_known_node_meets_constraint = (float('inf'), float('inf'), -1, -1)

        # logging
        self.log = None

    def __del__(self):
        if self.log:
            self.log.close()

    def reset(self):
        """
        resets the environment and returns the state
        """
        self.iteration = 0
        self.episode += 1
        self.node, self.levels = float('inf'), float('inf')
        self.sequence = ['strash']
        self.episode_dir = os.path.join(self.params['playground_dir'], str(self.episode))
        if not os.path.exists(self.episode_dir):
            os.makedirs(self.episode_dir)

        # logging
        log_file = os.path.join(self.episode_dir, 'log.csv')
        if self.log:
            self.log.close()
        self.log = open(log_file, 'w')
        self.log.write('iteration, optimization, LUT-6, Levels, best LUT-6 meets constraint, best LUT-6, best levels\n')

        normalize = Normalizer(self.state_dim)
        normalize.reset()

        state_node, state_level, _, graph = self._run()
        state = np.array([state_node, state_level])

        self._lastStats = state  # The initial AIG statistics
        self._curStats = self._lastStats

        self.lastAct = self.action_dim - 1
        self.lastAct2 = self.action_dim - 1
        self.lastAct3 = self.action_dim - 1
        self.lastAct4 = self.action_dim - 1

        combined = self.state(self.action_dim - 1, state_node, state_level)
        combined_state = torch.from_numpy(combined.astype(np.float32)).float()
        #combined_state = np.concatenate((state, state2), axis=-1)

        # logging
        self.log.write(
            ', '.join([str(self.iteration), self.sequence[-1], str(int(self.node)), str(int(self.levels))]) + '\n')
        self.log.flush()

        return (combined_state, graph)

    def step(self, optimization):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """
        self.sequence.append(self.params['optimizations'][optimization])
        new_node, new_level, reward, new_graph = self._run()

        state_statistics = self.state(optimization, new_node, new_level)
        state_statistics = torch.from_numpy(state_statistics.astype(np.float32)).float()
        # combined_state = np.concatenate((new_state, state2), axis=-1)

        # logging
        if self.node < self.best_known_node[0]:
            self.best_known_node = (int(self.node), int(self.levels), self.episode, self.iteration)
        if self.levels < self.best_known_levels[1]:
            self.best_known_levels = (int(self.node), int(self.levels), self.episode, self.iteration)
        if self.levels <= self.params['fpga_mapping']['levels'] and self.node < self.best_known_node_meets_constraint[
            0]:
            self.best_known_node_meets_constraint = (int(self.node), int(self.levels), self.episode, self.iteration)
        self.log.write(
            ', '.join([str(self.iteration), self.sequence[-1], str(int(self.node)), str(int(self.levels))]) + ', ' +
            '; '.join(list(map(str, self.best_known_node_meets_constraint))) + ', ' +
            '; '.join(list(map(str, self.best_known_node))) + ', ' +
            '; '.join(list(map(str, self.best_known_levels))) + '\n')
        self.log.flush()

        return (state_statistics, new_graph), reward, self.iteration == self.params['iterations'], None

    def _run(self):
        """
        run ABC on the given design file with the sequence of commands
        """
        self.iteration += 1
        output_design_file = os.path.join(self.episode_dir, str(self.iteration) + '.v')
        self.design = os.path.join(self.episode_dir, str(self.iteration - 1) + '.v')

        # output_design_file_mapped = os.path.join(self.episode_dir, str(self.iteration) + '-mapped.v')

        if self.iteration > 2:
            abc_command = 'read ' + self.params['mapping']['library_file'] + '; '
            abc_command += 'read_verilog ' + self.design + '; '
            # abc_command += 'read ' + self.params['design_file'] + '; '
            abc_command += 'strash' + '; '
            abc_command += ';'.join([self.sequence[-1], ]) + '; '
            # abc_command += 'self.sequence[-1]' + '; '
            abc_command += 'write ' + output_design_file + '; '
            # abc_command += 'if -K ' + str(self.params['fpga_mapping']['lut_inputs']) + '; '
            # abc_command += 'write ' + output_design_file_mapped + '; '
            abc_command += 'print_stats;'
        else:
            abc_command = 'read ' + self.params['mapping']['library_file'] + '; '
            abc_command += 'read ' + self.file + '; '
            abc_command += 'strash' + '; '
            abc_command += ';'.join(self.sequence) + '; '
            abc_command += 'write ' + output_design_file + '; '
            # abc_command += 'if -K ' + str(self.params['fpga_mapping']['lut_inputs']) + '; '
            # abc_command += 'write ' + output_design_file_mapped + '; '
            abc_command += 'print_stats;'

        # try:
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # get reward
        node, levels = self._get_metrics(proc)
        reward = self._get_reward(node, levels)
        self.node, self.levels = node, levels
        # get new state of the circuit
        state_node, state_level = self._get_state(output_design_file)

        # normalize = Normalizer(len(state))
        # normalize.reset()
        # normalize.observe(x=state)
        # nor_state_statistics = normalize.normalize(inputs=state)

        # generate circuit graph
        graph = read_verilog(output_design_file)
        return state_node, state_level, reward, graph

        """
        except Exception as e:
            print(e)
            print('遇到了故障')
            return None, None
        """

    def state(self, optimization, Numnode, Numlev):
        """
        Action and Markov ID
        """
        # state = np.array([Numnode, Numlev])

        self._lastStats[0] = self._curStats[0]
        self._curStats[0] = Numnode

        self._lastStats[1] = self._curStats[1]
        self._curStats[1] = Numlev

        stateArray = np.array([self._curStats[0] / self.initial_node, self._curStats[1] / self.initial_level,
                               self._lastStats[0] / self.initial_node, self._lastStats[1] / self.initial_level])

        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = optimization

        lastOneHotActs = np.zeros(self.action_dim)
        lastOneHotActs[self.lastAct2] += 1 / 3
        lastOneHotActs[self.lastAct3] += 1 / 3
        lastOneHotActs[self.lastAct] += 1 / 3
        Markov_ID = np.array([float(self.iteration - 1) / self.total_len_seq])
        combined = np.concatenate((stateArray, lastOneHotActs, Markov_ID), axis=-1)

        return combined

    def _get_metrics(self, stats):
        """
        parse LUT node and levels from the stats command of ABC
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()

        ob = re.search(r'lev *= *[0-9]+', line)
        levels = int(ob.group().split('=')[1].strip())

        ob = re.search(r'and *= *[0-9]+', line)
        node = int(ob.group().split('=')[1].strip())

        return node, levels

    def _get_reward(self, cur_node, cur_levels):
        # now calculate the reward
        #log('initial_node: ' + str(self.initial_node))
        log('resyn2_node: ' + str(self.resyn2_node))
        log('current_node: ' + str(cur_node))
        #log('last_node: ' + str(self.node))
        #reward = math.exp((self.initial_node - cur_node) / self.initial_node) - 1
        #reward = (self.node - cur_node) / float(self.initial_node) - self.reward_baseline
        reward = (self.resyn2_node - cur_node) / self.resyn2_node
        log('reward: ' + str(reward))
        #log('reward_baseline: ' + str(self.reward_baseline))

        return reward  # self._reward_table(constraint_met, constraint_improvement, optimization_improvement)

    def _get_state(self, design_file):
        return extract_features(design_file, self.params['yosys_binary'], self.params['abc_binary'])

    def curStatsValue(self):
        return float(self.node) / float(self.initial_node)
    
    def returns(self):
        return [self.node, self.levels]


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
