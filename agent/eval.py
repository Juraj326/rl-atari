'''
This file contains the evaluation script used for the results found in Appendix C.
The script expects the directory given in `path` to contain only agents with the same hyperparameters (eg. SD-SAC w/ 0.5 beta & 0.5 c).
'''

import os
from dataclasses import dataclass
import tyro
import numpy as np
from utils import load_model, vector_env_setup, choose_device, evaluate_agent
from train import Args
from networks import Actor

@dataclass
class EvalArgs:
    path: str = tyro.MISSING
    """the path to the directory of the saved models"""
    runs: int = 5
    """the number of evaluation runs"""
    episodes: int = 50
    """the number of episodes in an evaluation run"""
    num_envs: int = 8
    """the number of parallel environments running"""
    backend: str = 'cpu'
    """which backend to use: `mps` or `cuda`"""

if __name__ == "__main__":
# ================ Setup ================
    eval_args = tyro.cli(EvalArgs)

    if eval_args.num_envs < 1:
        raise ValueError('Number of environments must be at least 1.')
    if not os.path.isdir(eval_args.path):
        raise ValueError('Please provide a directory to the saved models.')

    model_files = list()
    for file in os.listdir(eval_args.path):
        if os.path.isfile(os.path.join(eval_args.path, file)) and file.endswith('.pt'):
            model_files.append(os.path.join(eval_args.path, file))

    if len(model_files) < 1:
        raise ValueError(f'Could not find any models in {eval_args.path}.')

    device = choose_device(eval_args.backend)
    total_rewards = np.zeros(len(model_files) * eval_args.runs * eval_args.episodes)
    total_lengths = np.zeros(len(model_files) * eval_args.runs * eval_args.episodes)
    episode_count = 0

    for i, model_file in enumerate(model_files):
        print(f'Evaluating model {model_file}')
        loaded_model = load_model(model_file, device)
        args = loaded_model['args']
        args.num_envs = eval_args.num_envs
        run_name = f'eval_{model_file[:-3]}'
        envs = vector_env_setup(args, run_name, True)

        actor = Actor(envs).to(device)
        actor.load_state_dict(loaded_model['actor'])
        rewards, lengths = evaluate_agent(actor, eval_args.runs, eval_args.episodes, envs, device, True)

        envs.close()

        start = i * eval_args.runs * eval_args.episodes
        end = (i + 1) * eval_args.runs * eval_args.episodes
        total_rewards[start:end] += rewards
        total_lengths[start:end] += lengths

    print(f'Rewards:\n{total_rewards}')
    print(f'Lengths:\n{total_lengths}')
    print(f'Mean episodic reward: {total_rewards.mean():.2f}')
    print(f'Best achieved reward: {total_rewards.max():.2f}')
    print(f'Mean episodic length: {total_lengths.mean():.2f}')
    print(f'Longest game: {total_lengths.max():.2f}')