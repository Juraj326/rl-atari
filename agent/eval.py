'''
This file contains the evaluation script used for the results found in Appendix B.
The script expects the directory given in `path` to contain only agents with the same hyperparameters (eg. SAC+AE).
'''

import os
import random
from dataclasses import dataclass
import tyro
import torch
import numpy as np
from utils import load_model, vector_env_setup, get_episodic_stats
from train import Args
from networks import Actor, Encoder

@dataclass
class EvalArgs:
    path: str = tyro.MISSING
    """the path to the directory of the saved model"""
    runs: int = 5
    """the number of evaluation runs"""
    episodes: int = 50
    """the number of episodes in an evaluation run"""
    num_envs: int = 8
    """the number of parallel environments running"""

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

    device = torch.device('cpu')
    seeds = list([random.randint(1, 10_000) for _ in range(eval_args.runs)])
    total_rewards = np.zeros(len(model_files) * eval_args.runs * eval_args.episodes)
    total_lengths = np.zeros(len(model_files) * eval_args.runs * eval_args.episodes)
    episode_count = 0

# ================ Model and environment setup ================
    for model_file in model_files:
        print(f'Evaluating model {model_file}')
        loaded_model = load_model(model_file, device)
        args = loaded_model['args']
        args.num_envs = eval_args.num_envs
        run_name = f'eval_{eval_args.path[:-3]}'
        num_episodes = eval_args.episodes
        for run in range(eval_args.runs):
            args.seed = seeds[run]
            envs = vector_env_setup(args, run_name, True)
            encoder = Encoder(envs).to(device)
            encoder.load_state_dict(loaded_model['encoder'])
            actor = Actor(envs, encoder).to(device)
            actor.load_state_dict(loaded_model['actor'])
            encoder.eval()
            actor.eval()
            episodes = 0
            obs, infos = envs.reset(seed=seeds[run])
            print(f'Evaluation run {run + 1}, seed: {args.seed}')

# ================ Evaluation loop ================
            while episodes < num_episodes:
                with torch.inference_mode():
                    actions = actor.get_action(torch.from_numpy(obs).to(device))
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                if terminations.any() or truncations.any():
                    episodic_rewards, episodic_lengths = get_episodic_stats(infos)
                    for reward, length in zip(episodic_rewards, episodic_lengths):
                        if episodes >= num_episodes:
                            break

                        total_rewards[episode_count] = reward
                        total_lengths[episode_count] = length
                        print(f'Episode {episodes + 1} reward: {reward}')
                        print(f'Episode {episodes + 1} length: {length}')
                        episodes += 1
                        episode_count += 1

                obs = next_obs

            envs.close()

    print(f'Rewards:\n{total_rewards}')
    print(f'Lengths:\n{total_lengths}')
    print(f'Mean episodic reward: {total_rewards.mean():.2f}')
    print(f'Best achieved reward: {total_rewards.max():.2f}')
    print(f'Mean episodic length: {total_lengths.mean():.2f}')
    print(f'Longest game: {total_lengths.max():.2f}')