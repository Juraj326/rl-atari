'''
This file contains various utility methods.
The environment setup method `make_env` and target network update method `update_target_networks` were refactored,
and moved here from the training script, original: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
'''

import os
import glob
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, AutoresetMode
import ale_py
import torch
import numpy as np
import wandb
from collections import namedtuple
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from wrappers import InverseReward, RescaleReward, LifeLostPenalty

Model = namedtuple('Model', ['encoder_sd', 'encoder_target_sd', 'decoder_sd', 'actor_sd', 'critic1_sd', 'critic2_sd', 'target1_sd', 'target2_sd', 'actor_optimizer_sd', 'critic_optimizer_sd', 'log_alpha', 'alpha_optimizer_sd', 'global_step', 'args'])

def choose_device(backend: str):
    if backend == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif backend == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')

    return torch.device('cpu')

def apply_reward_wrapper(env, reward_wrapper: str):
    if reward_wrapper == 'clipped':
        return ClipRewardEnv(env)
    elif reward_wrapper == 'score':
        return RescaleReward(env, 0.1)
    elif reward_wrapper == 'inverse':
        env = InverseReward(env)
        return RescaleReward(env, 0.1)

    raise ValueError(f'Invalid reward wrapper: {reward_wrapper}')
    return env

def make_env(args, idx: int, run_name: str, eval: bool):
    def thunk():
        if not eval and args.capture_video and idx == 0:
            env = gym.make(args.env_id, render_mode='rgb_array')
            capture_frequency = 25  # every 25th episode
            env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda x: (x > 0) and x % capture_frequency == 0, fps=60, disable_logger=False)
        else:
            env = gym.make(args.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if not eval and args.episodic_life:
            env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        if not eval:
            env = apply_reward_wrapper(env, args.reward_wrapper)

        if not eval and args.life_loss_penalty:
            env = LifeLostPenalty(env, -1.0)

        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)

        env.action_space.seed(args.seed + idx)

        assert isinstance(env.action_space, gym.spaces.Discrete), 'only discrete action space is supported'
        return env

    return thunk

def vector_env_setup(args, run_name: str, eval: bool = False):
    if args.num_envs < 1:
        raise ValueError('Number of environments must be at least 1')
    if args.num_envs == 1:
        return SyncVectorEnv([make_env(args, 0, run_name, eval)], autoreset_mode=AutoresetMode.SAME_STEP)

    return AsyncVectorEnv([make_env(args, idx, run_name, eval) for idx in range(args.num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)

def get_episodic_stats(infos: dict):
    info = infos['final_info'].get('episode')
    mask = np.where(infos['final_info'].get('_episode'))[0]
    rewards = list()
    lengths = list()

    if info is not None:
        for idx in mask:
            rewards.append(float(info['r'][idx]))
            lengths.append(float(info['l'][idx]))

    return rewards, lengths

def update_target_networks(network, target, tau: float, encoders: bool):
    if encoders:
        network_params = network.parameters()
        target_params = target.parameters()
    else:
        network_params = list(network.fc1.parameters()) + list(network.fc2.parameters()) + list(network.fc_q.parameters())
        target_params = list(target.fc1.parameters()) + list(target.fc2.parameters()) + list(target.fc_q.parameters())

    for param, target_param in zip(network_params, target_params):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def save_model(model: Model, file_name: str):
    os.makedirs('models', exist_ok=True)

    checkpoint = {
        'encoder': model.encoder_sd,
        'encoder target': model.encoder_target_sd,
        'decoder': model.decoder_sd,
        'actor': model.actor_sd,
        'critic1': model.critic1_sd,
        'critic2': model.critic2_sd,
        'target1': model.target1_sd,
        'target2': model.target2_sd,
        'actor optimizer': model.actor_optimizer_sd,
        'critic optimizer': model.critic_optimizer_sd,
        'log alpha': model.log_alpha,
        'alpha optimizer': model.alpha_optimizer_sd,
        'global step': model.global_step,
        'args': model.args
    }

    path = f'models/{file_name}.pt'
    torch.save(checkpoint, path)
    print(f'Model has been saved at: {path}')
    return path

def load_model(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No model found at {path}.')

    model = torch.load(path, map_location=device, weights_only=False)
    print(f'Model loaded, trained for {model["global step"]} steps.')
    return model

def upload_videos(run_name: str, env_id: str):
    video_dir = os.path.join('videos', run_name)
    if os.path.exists(video_dir):
        video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
        for video_file in video_files:
            base_name = os.path.basename(video_file)
            wandb.log(
                {
                    f'media/{base_name}': wandb.Video(
                        video_file,
                        caption=env_id,
                        format='mp4'
                    )
                },
            )
            print(f'Uploaded {video_file} to WandB.')

def upload_model(run_name: str, path: str):
    if os.path.exists(path):
        artifact = wandb.Artifact(name=f'{run_name}', type='model')
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        print(f'Uploaded {path} to WandB.')