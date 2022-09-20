import os
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
#from .scl_session import SCLSession as SCLGame
from .environment import LogicSession as SCLGame
import yaml
import argparse
from dgl.nn.pytorch import GraphConv
import dgl
import csv
import torch.nn.functional as F
from torchsummary import summary
from .training_file import AgentDiscretePPO
from .training_file import Arguments as args
import random
random.seed(1)
import numpy as np


design = '/home/yangch/EPFL/random_control/ctrl.aig'#'/home/yangch/abc/i10.aig'


def log(message):
    print('[LSJedi {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)


def inference(options):
    torch.manual_seed(1)
    np.random.seed(1)
    file = design
    env = SCLGame(options, file)
    model = torch.load(options['model_dir'])

    actor = model.act
    critic = model.cri

    #pre_value1 = pre_value2 = pre_value3 = None

    actor.eval()
    critic.eval()
    state = env.reset()

    if_quit = True
    get_pre_value = 0
    while if_quit:
        with torch.no_grad():
            state_statistics = state[0]
            state_statistics = torch.as_tensor((state_statistics,), dtype=torch.float32)
            graph = state[1]
            action, a_prob = actor.get_action(state_statistics, graph)
            print('inference_action_prob---->', a_prob)
            action = torch.argmax(a_prob)
            print('inference_action---->', action)
            pre_value = critic(state_statistics, graph)
            next_state, reward, done, _ = env.step(action)
            state = next_state

            log('pre_value: ' + str(pre_value))

            if get_pre_value < 1:
                #pre_value3 = pre_value2 = pre_value1 = pre_value
                pre_value2 = pre_value1 = pre_value
            get_pre_value += 1

        if pre_value1 < pre_value2 and pre_value < pre_value1 and pre_value2 < pre_value3:
            if_quit = False

        pre_value3 = pre_value2
        pre_value2 = pre_value1
        pre_value1 = pre_value

