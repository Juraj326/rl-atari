'''
Used to make Figures `Atari 2600 Space Invaders in the ALE collection of environments` and `Visualization of an observation`.
'''

import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np

# Atari 2600 Space Invaders in the ALE collection of environments
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
observation, _ = env.reset()

start = 30 # prvych 30 env stepov sa neda hybat
ufo = 300 # na 300 env stepe sa objavi command spaceship
for i in range(start + 6):
    env.step(2)

for i in range(start + 6, ufo):
    observation, _, _, _, _ = env.step(0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(observation)
ax.set_title('Atari 2600 Space Invaders in the ALE collection of environments')
ax.axis('off')
plt.show()
env.close()

# Visualization of an observation
env_id = 'SpaceInvadersNoFrameskip-v4'
env = gym.make(env_id, render_mode='rgb_array')
env_p = gym.wrappers.ResizeObservation(env, (84, 84))
env_p = gym.wrappers.GrayscaleObservation(env_p)
env_p = gym.wrappers.FrameStackObservation(env_p, 4)

observation, _ = env.reset()
observation_p, _ = env_p.reset()
for _ in range(10):
    observation, _, _, _, _ = env.step(0)
    observation_p, _, _, _, _ = env_p.step(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(observation)
ax1.set_title('Visualization of the unprocessed observation')
ax1.axis('off')

composite = np.zeros([84, 84])
for i in range(3):
    composite += observation_p[i]

composite /= 255.0

ax2.imshow(composite, cmap='gray')
ax2.set_title('Visualization of the preprocessed observation')
ax2.axis('off')

plt.tight_layout()
plt.show()