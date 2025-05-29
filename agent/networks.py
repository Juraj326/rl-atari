'''
This file contains refactored and modified definitions of the agent's neural networks according to the SAC+AE paper.
Original by CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Orthogonal
def lin_layer_init(layer, bias_const=0.0):
    nn.init.orthogonal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# Delta orthogonal
def conv_layer_init(layer, bias_const=0.0):
    # https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py#L38
    layer.weight.data.fill_(0.0)
    layer.bias.data.fill_(0.0)
    mid = layer.weight.size(2) // 2
    gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(layer.weight.data[:, :, mid, mid], gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Encoder(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.feature_dim = 50

        self.conv = nn.Sequential(
            conv_layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            conv_layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            conv_layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            conv_layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc = lin_layer_init(nn.Linear(self.output_dim, self.feature_dim))
        self.normalize = nn.LayerNorm(self.feature_dim)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv(x))
        x = self.fc(x)
        latent_vector = F.tanh(self.normalize(x))
        return latent_vector

class Decoder(nn.Module):
    def __init__(self, envs, feature_dim, output_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        obs_shape = envs.single_observation_space.shape
        self.spatial_size = int((output_dim / 32) ** 0.5)

        self.fc = lin_layer_init(nn.Linear(feature_dim, output_dim))

        self.conv = nn.Sequential(
            conv_layer_init(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            conv_layer_init(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            conv_layer_init(nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            conv_layer_init(nn.ConvTranspose2d(32, obs_shape[0], kernel_size=3, stride=2, output_padding=1)),
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 32, self.spatial_size, self.spatial_size)
        obs = self.conv(x)
        return obs

class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = lin_layer_init(nn.Linear(self.encoder.feature_dim, 1024))
        self.fc2 = lin_layer_init(nn.Linear(1024, 1024))
        self.fc_q = lin_layer_init(nn.Linear(1024, envs.single_action_space.n))

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_vals = self.fc_q(x)
        return q_vals

class Actor(nn.Module):
    def __init__(self, envs, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = lin_layer_init(nn.Linear(self.encoder.feature_dim, 1024))
        self.fc2 = lin_layer_init(nn.Linear(1024, 1024))
        self.fc_logits = lin_layer_init(nn.Linear(1024, envs.single_action_space.n))

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)
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