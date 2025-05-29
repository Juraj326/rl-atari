'''
This file contains various reward wrappers for the SpaceInvaders ALE environment.
'''

from gymnasium import RewardWrapper, Env
from typing import SupportsFloat

class InverseReward(RewardWrapper):
    # https://atariage.com/manual_html_page.php?SoftwareLabelID=460
    def __init__(self, env: Env):
        super().__init__(env)
        self.base = 30.0 + 5.0  # 30 = max reward + 5 for inverse

    def reward(self, reward: SupportsFloat) -> float:
        reward = float(reward)
        if reward == 0.0:
            return reward

        return self.base - reward

class RescaleReward(RewardWrapper):
    def __init__(self, env: Env, scale: float):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: SupportsFloat) -> float:
        return float(reward) * self.scale

class LifeLostPenalty(RewardWrapper):
    def __init__(self, env: Env, penalty: float):
        super().__init__(env)
        self.env = env
        self.penalty = penalty
        self.lives = 0

    def reward(self, reward: SupportsFloat) -> float:
        current_lives = self.env.unwrapped.ale.lives()
        reward = float(reward)
        if current_lives < self.lives:
            reward += self.penalty
        elif self.lives == 1 and current_lives > 1: # check whether the agent has lost its final life
            reward += self.penalty

        self.lives = current_lives
        return reward