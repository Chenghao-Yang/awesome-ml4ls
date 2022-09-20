import yaml
import os
import argparse
import datetime
import numpy as np
import time
from lpl.training_file import logic_jedi_master
from pyfiglet import Figlet
from lpl.inference import inference 


def log(message):
    print('[LSJedia {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs logic synthesis optimization using RL')

    parser.add_argument("params", type=open, nargs='?', default='params.yml', \
                        help="Path to the params.yml file")
    parser.add_argument("mode", type=str, choices=['train', 'optimize'], \
        help="Use the design to train the model or only optimize it")

    args = parser.parse_args()
    options = yaml.load(args.params, Loader=yaml.FullLoader)
    # learner = demo_discrete_action(options=options)

    f = Figlet(font='slant')
    print(f.renderText('LSJedi'))
    
    if args.mode == 'train':
        log('Starting to train the agent ..')

        dir = options['playground_dir']
        if not os.path.exists(dir):
            os.makedirs(dir)
    
        logic_jedi_master(options)
        
    elif args.mode == 'optimize':
        log('Starting agent to optimize')
        inference(options)