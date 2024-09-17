import os
import json
import random
import argparse
# import numpy as np
# import torch
from methods import run_naive

# Settings of the project
with open('./.openai-config.json') as config_file:
    openai_config = json.load(config_file)
    os.environ["OPENAI_API_KEY"] = openai_config['openai_api_key']
    os.environ["OPENAI_BASE_URL"] = openai_config['openai_base_url']

def run(config):
    if config['method'] == 'naive':
        run_naive(config)
    else:
        raise NotImplementedError('unknown conjecture method.')

def main():
    parser = argparse.ArgumentParser(description="This program tries to create new math questions from old datasets.")
    parser.add_argument('-c', '--config', type=str, help='path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1145, help='random seed for the program')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='The temperature parameter for inference')

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
        config['seed'] = args.seed
        config['temperature'] = args.temperature
        random.seed(args.seed)
        run(config)
    print(f"program run with seed {args.seed}")

if __name__ == "__main__":
    main()
