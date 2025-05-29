'''
This file contains the refactored and modified training script.
For the base, we used the CleanRL sac_atari.py implementation of SAC-Discrete: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
'''

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard.writer import SummaryWriter

from utils import choose_device, vector_env_setup, save_model, upload_videos, Model, get_episodic_stats, update_target_networks, sync_encoders, upload_model, encode, package_embeddings, evaluate_agent
from networks import SoftQNetwork, Actor
from memory import ReplayBuffer, EmbeddingsReplayBuffer

@dataclass
class Args:
# ================ Miscellaneous arguments ================
    exp_name: str = ''
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = 'SAC+extensions'
    """the wandb's project name"""
    wandb_entity: str = ''
    """the entity (team) of wandb's project"""
    wandb_group: str = ''
    """the group in the wandb project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    evaluate: bool = False
    """whether to evaluate the agent's deterministic policy during training"""
    eval_frequency: int = 50_000
    """the frequency of evaluation runs"""
    eval_runs: int = 3
    """the number of evaluation runs to perform"""
    eval_episodes: int = 10
    """the number of episodes in each evaluation run"""
    backend: str = 'cuda'
    """which backend to use: `mps` or `cuda`"""
    save_model: bool = True
    """whether to save the model at `./models/`"""

# ================ Algorithm specific arguments ================
    env_id: str = 'SpaceInvadersNoFrameskip-v4'
    """the id of the environment"""
    total_timesteps: int = 5_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 500_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient"""
    batch_size: int = 64
    """the batch size of sample from the replay memory"""
    learning_starts: int = 20_000
    """timestep to start learning"""
    policy_lr: float = 0.0003
    """the learning rate of the policy network optimizer"""
    q_lr: float = 0.0003
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8_000
    """the frequency of updates for the target networks"""
    alpha: float = 0.05
    """entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""

# ================ SD-SAC specific arguments ================
    entropy_penalty: bool = False
    """use entropy penalty"""
    beta: float = 0.5
    """entropy-penalty coefficient"""
    double_avg_w_q_clip: bool = False
    """use double average Q-learning with Q-clip"""
    c: float = 0.5
    """Q-clip range"""

# ================ Munchausen RL specific arguments ================
    munchausen: bool = False
    """whether to use Munchausen RL"""
    reward_scale: float = 0.9
    """munchausen reward scaling factor"""
    l0: float = -1
    """log-policy clip range"""

# ================ SEER specific arguments ================
    freeze_encoders: bool = False
    """whether to freeze encoders (SEER)"""
    store_embeddings: bool = False
    """whether to store embeddings (SEER)"""
    tf: int = 200_000
    """network updates to activate SEER"""

# ================ Environment specific arguments ================
    episodic_life: bool = True
    """whether to use the `EpisodicLifeEnv` wrapper (treat each life as single episode)"""
    life_loss_penalty: bool = False
    """whether to penalize the agent when it loses a life"""
    reward_wrapper: str = 'clipped'
    """which reward wrapper to use: `clipped`, `score` or `inverse`"""
    num_envs: int = 4
    """the number of parallel environments running"""

if __name__ == "__main__":
# ================ SETUP ================
    args = tyro.cli(Args)

    if not args.freeze_encoders and args.store_embeddings:
        raise ValueError('To store embeddings you must also enable encoder freezing.')

    run_name = f"{args.env_id}__{args.seed}__{int(time.time())}"
    if args.exp_name != '':
        run_name = args.exp_name

    # Determine device
    device = choose_device(args.backend)
    print(f'Using device: {device}')

    # Setup WandB
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            resume='allow',
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Environment setup
    envs = vector_env_setup(args, run_name)
    eval_envs = vector_env_setup(args, run_name, True)

    # Initialize networks
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)

    # Equalise target and local network weights
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Initialize optimizers
    # eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    # Initialize an empty replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    # Start the game
    obs, _ = envs.reset(seed=args.seed)
    using_seer = args.freeze_encoders or args.store_embeddings
    seer_active = False
    mid_training = False
    timing = False
    segment_start = args.learning_starts  # for timing
    global_step = 0
    learning_starts = global_step + args.learning_starts

# ================ Training loop ================
    while global_step < args.total_timesteps:
        input_obs = torch.Tensor(obs).to(device)
        if actor.using_embeddings:
            # Use embeddings as input data (SEER)
            actor_emb, qf1_emb, qf2_emb = encode(input_obs, actor, qf1, qf2)
            embeddings = package_embeddings(actor_emb, qf1_emb, qf2_emb)
            input_data = (actor_emb, qf1_emb, qf2_emb)
        else:
            input_data = input_obs

    # ================ Data collection ================
        # Action selection
        if not mid_training and global_step < learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            # Entropy for uniformly sampled actions is log(n) (SD-SAC)
            entropies = np.full(envs.num_envs, np.log(np.prod(envs.single_action_space.n)))
        else:
            # Sample action from the policy
            actions, log_prob, action_probs, entropy = actor.get_action(input_data)
            actions = actions.detach().cpu().numpy()
            entropies = entropy.detach().cpu().numpy()

        # Sample transition from the environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log end of episode statistics (stochastic policy)
        if 'final_info' in infos and '_episode' in infos['final_info']:
            episodic_rewards, episodic_lengths = get_episodic_stats(infos)
            for episodic_reward, episodic_length in zip(episodic_rewards, episodic_lengths):
                writer.add_scalar('charts/episodic_return', episodic_reward, global_step)
                writer.add_scalar('charts/episodic_length', episodic_length, global_step)

        # Handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        # Store the transition in the replay buffer
        if actor.using_embeddings:
            # Store embeddings (SEER)
            input_real_next_obs = torch.Tensor(real_next_obs).to(device)
            actor_next_emb, qf1_next_emb, qf2_next_emb = encode(input_real_next_obs, actor, qf1, qf2)
            next_embeddings = package_embeddings(actor_next_emb, qf1_next_emb, qf2_next_emb)
            rb.add(embeddings, next_embeddings, actions, rewards, entropies, terminations, infos)
        else:
            rb.add(obs, real_next_obs, actions, rewards, entropies, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    # ================ Training ================
        if global_step > learning_starts:
            if not timing:
                start_time = time.time()
                segment_start = global_step
                timing = True

            previous_step = global_step - args.num_envs
            num_gradient_steps = (global_step // args.update_frequency) - (previous_step // args.update_frequency)
            for _ in range(num_gradient_steps):
                data = rb.sample(args.batch_size)
                if actor.using_embeddings:
                    # Use embeddings as input data (SEER)
                    input_data = data.embeddings
                    input_data_next = data.next_embeddings
                else:
                    input_data = data.observations
                    input_data_next = data.next_observations

        # ================ Critic training ================
                with torch.no_grad():
                    # Target Q-value estimation
                    _, next_state_log_pi, next_state_action_probs, _ = actor.get_action(input_data_next)
                    qf1_next_target = qf1_target(input_data_next, 1)
                    qf2_next_target = qf2_target(input_data_next, 2)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    # adapt Q-target for discrete Q-function
                    if args.double_avg_w_q_clip:
                        # Double average Q-Value estimation (SD-SAC)
                        qf_estimates = torch.mean(torch.stack([qf1_next_target, qf2_next_target], dim=0), dim=0)
                    else:
                        qf_estimates = torch.min(qf1_next_target, qf2_next_target)

                    qf_next_target = next_state_action_probs * (qf_estimates - alpha * next_state_log_pi)
                    qf_next_target = qf_next_target.sum(dim=1)

                    reward_term = data.rewards.flatten()
                    if args.munchausen:
                        # Augment the reward them (Munchausen RL)
                        _, log_pi, _, _ = actor.get_action(input_data)
                        munchausen_log_pi = log_pi.gather(1, data.actions.long()).view(-1)
                        # In the paper, tau is the temperature parameter (alpha in our case) and alpha is the scaling factor
                        reward_term = reward_term + args.reward_scale * torch.clamp(alpha * munchausen_log_pi, min=args.l0, max=0)

                    next_q_value = reward_term + (1 - data.dones.flatten()) * args.gamma * (qf_next_target)

                # Update the Q-function parameters
                # use Q-values only for the taken actions
                qf1_values = qf1(input_data, 1)
                qf2_values = qf2(input_data, 2)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

                if args.double_avg_w_q_clip:
                    # Q-clipping (SD-SAC)
                    with torch.no_grad():
                        qf1_old_target = qf1_target(input_data, 1)
                        qf2_old_target = qf2_target(input_data, 2)
                    qf1_target_a_values = qf1_old_target.gather(1, data.actions.long()).view(-1)
                    qf2_target_a_values = qf2_old_target.gather(1, data.actions.long()).view(-1)
                    # Clip the Q-value estimation from target critic network 1
                    qf1_clipped = qf1_target_a_values + torch.clamp(qf1_a_values - qf1_target_a_values, -args.c, args.c)
                    qf1_clipped_loss = F.mse_loss(qf1_clipped, next_q_value)
                    qf1_loss = torch.max(qf1_loss, qf1_clipped_loss)
                    # Clip the Q-value estimation from target critic network 2
                    qf2_clipped = qf2_target_a_values + torch.clamp(qf2_a_values - qf2_target_a_values, -args.c, args.c)
                    qf2_clipped_loss = F.mse_loss(qf2_clipped, next_q_value)
                    qf2_loss = torch.max(qf2_loss, qf2_clipped_loss)

                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

        # ================ Actor training ================
                _, log_pi, action_probs, entropy_pi = actor.get_action(input_data)
                qf1_values = qf1_values.detach()
                qf2_values = qf2_values.detach()
                if args.double_avg_w_q_clip:
                    # Double average Q-Value estimation (SD-SAC)
                    qf_values = torch.mean(torch.stack([qf1_values, qf2_values], dim=0), dim=0)
                else:
                    qf_values = torch.min(qf1_values, qf2_values)

                # Update policy weights
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - qf_values)).mean()
                if args.entropy_penalty:
                    # Retrieve the entropy term H_{pi_old} from the replay buffer and calculate entropy penalty (SD-SAC)
                    entropy_pi_old = data.entropies.detach()
                    entropy_penalty = args.beta * (1/2) * F.mse_loss(entropy_pi_old, entropy_pi)
                    actor_loss = actor_loss + entropy_penalty

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

        # ================ Update temperature ================
                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # ================ Update target networks ================
            if (global_step // args.target_network_frequency) > (previous_step // args.target_network_frequency):
                update_target_networks(qf1, qf1_target, args.tau, seer_active and args.freeze_encoders)
                update_target_networks(qf2, qf2_target, args.tau, seer_active and args.freeze_encoders)

        # ================ Performance metrics logging ================
            sps = int((global_step - segment_start) / (time.time() - start_time))
            if (global_step // 10_000) > (previous_step // 10_000):
                print(f'SPS={sps}, global_step={global_step}')  # Print steps per second and current step every 10,000 steps
            if global_step % 100 < args.num_envs:
                writer.add_scalar('losses/qf1_values', qf1_a_values.mean().item(), global_step)
                writer.add_scalar('losses/qf2_values', qf2_a_values.mean().item(), global_step)
                writer.add_scalar('losses/qf1_loss', qf1_loss.item(), global_step)
                writer.add_scalar('losses/qf2_loss', qf2_loss.item(), global_step)
                writer.add_scalar('losses/qf_loss', qf_loss.item() / 2.0, global_step)
                writer.add_scalar('losses/actor_loss', actor_loss.item(), global_step)
                writer.add_scalar('losses/alpha', alpha, global_step)
                writer.add_scalar('losses/entropy_pi', entropy_pi.mean().item(), global_step)
                if args.entropy_penalty:
                    writer.add_scalar('losses/entropy_pi_old', entropy_pi_old.mean().item(), global_step)
                    writer.add_scalar('losses/entropy_penalty', entropy_penalty.item(), global_step)
                writer.add_scalar('charts/SPS', sps, global_step)
                if args.autotune:
                    writer.add_scalar('losses/alpha_loss', alpha_loss.item(), global_step)

        # ================ Agent evaluation (deterministic policy) ================
            if args.evaluate and (global_step // args.eval_frequency) > (previous_step // args.eval_frequency):
                print(f'Evaluating agent at timestep {global_step}')
                eval_reward, eval_length = evaluate_agent(actor, args.eval_runs, args.eval_episodes, eval_envs, device)
                print(f'Reward: {eval_reward.mean():.2f}, Length: {eval_length.mean():.2f}')
                writer.add_scalar('charts/dp_episodic_return', eval_reward.mean(), global_step)
                writer.add_scalar('charts/dp_episodic_length', eval_length.mean(), global_step)
                timing = False

        # ================ Activate SEER after T_f updates ================
            if using_seer and not seer_active and (global_step // args.update_frequency) >= args.tf:
                # Freeze encoders
                if args.freeze_encoders:
                    actor.freeze_encoder()
                    qf1.freeze_encoder()
                    qf2.freeze_encoder()
                    qf1_target.freeze_encoder()
                    qf2_target.freeze_encoder()

                    # Sync encoders
                    sync_encoders(qf1, qf1_target)
                    sync_encoders(qf2, qf2_target)
                    print(f'Freezing encoders at timestep {global_step}')

                # Store embeddings
                if args.store_embeddings:
                    actor.use_embeddings()
                    qf1.use_embeddings()
                    qf2.use_embeddings()
                    qf1_target.use_embeddings()
                    qf2_target.use_embeddings()

                    # Restart data collection phase
                    learning_starts += global_step

                    # Adjust buffer size for SEER replay buffer
                    P = 4 * 84 * 84 # 4 frames * 84x84 pixels
                    N = 3   # Actor, Critic1, Critic2 encoders
                    K = 1   # No data augmentation
                    L = actor.encoder.output_dim # Embedding size
                    buffer_size = int(args.buffer_size * (P / (4 * N * K * L)))
                    print(f'Embeddings buffer size: {buffer_size}')

                    # Free up memory used by the old replay buffer
                    rb = None
                    import gc
                    gc.collect()

                    # Initialize an empty replay buffer that stores embeddings
                    rb = EmbeddingsReplayBuffer(
                        buffer_size,
                        envs.single_observation_space,
                        envs.single_action_space,
                        device,
                        n_envs=args.num_envs,
                        handle_timeout_termination=False,
                        embedding_dim=actor.encoder.output_dim
                    )

                    # Sample actions from policy in 2nd transition collection phase
                    mid_training = True

                    print(f'Storing embeddings from timestep {global_step}')

                    # Reset timer to correctly measure SPS
                    timing = False

                seer_active = True

        global_step += args.num_envs

# ================ Post-training ================
    # Save the model after training has concluded
    if args.save_model:
        a_optimizer_sd = None
        if args.autotune:
            a_optimizer_sd = a_optimizer.state_dict()
        FinalModel = Model(
            actor.state_dict(),
            qf1.state_dict(),
            qf2.state_dict(),
            qf1_target.state_dict(),
            qf2_target.state_dict(),
            actor_optimizer.state_dict(),
            q_optimizer.state_dict(),
            alpha,
            a_optimizer_sd,
            global_step,
            args
        )
        path = save_model(FinalModel, run_name)

    # Upload data to WandB
    if args.track:
        if args.capture_video:
            upload_videos(run_name, args.env_id)
        if args.save_model:
            upload_model(run_name, path)

    envs.close()
    eval_envs.close()
    writer.close()