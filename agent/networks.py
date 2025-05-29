'''
This file contains refactored and modified definitions of the agent's neural networks.
Original by CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Encoder(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.frozen = False
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

    def forward(self, x):
        x = x / 255.0
        if self.frozen:
            with torch.no_grad():
                return F.relu(self.conv(x))

        return F.relu(self.conv(x))

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.frozen = True

class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.using_embeddings = False
        self.encoder = Encoder(envs)
        self.fc1 = layer_init(nn.Linear(self.encoder.output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x, i: int):
        if self.using_embeddings and self.training:
            embedding = x[i]
        else:
            embedding = self.encoder(x)

        latent_vector = F.relu(self.fc1(embedding))
        q_vals = self.fc_q(latent_vector)
        return q_vals

    def freeze_encoder(self):
        self.encoder.freeze()

    def use_embeddings(self):
        self.using_embeddings = True

class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.using_embeddings = False
        self.encoder = Encoder(envs)
        self.fc1 = layer_init(nn.Linear(self.encoder.output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        if self.using_embeddings and self.training:
            embedding = x[0]
        else:
            embedding = self.encoder(x)

        latent_vector = F.relu(self.fc1(embedding))
        logits = self.fc_logits(latent_vector)
        return logits

    def get_action(self, x):
        logits = self(x)
        if self.training:   # Sample action from the policy (training)
            policy_dist = Categorical(logits=logits)
            action = policy_dist.sample()
            # Action probabilities for calculating the adapted soft-Q loss
            action_probs = policy_dist.probs
            # Entropy term for calcuating the entropy penalty (SD-SAC)
            entropy = policy_dist.entropy()
            log_prob = F.log_softmax(logits, dim=1)
            return action, log_prob, action_probs, entropy
        else:   # Use deterministic action selection (evaluation)
            action = torch.argmax(logits, dim=1)
            return action

    def freeze_encoder(self):
        self.encoder.freeze()

    def use_embeddings(self):
        self.using_embeddings = True