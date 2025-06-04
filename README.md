# Reinforcement Learning in the domain of games

As the basis for our project, we used [CleanRL](https://github.com/vwxyzjn/cleanrl)'s `sac_atari.py` implementation of the SAC-Discrete algorithm. Environment simulation is handled by the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) framework, integrated with the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment interface.

## Project Structure

```sh
.
├── agent
│    ├── eval.py                # The evaluation script
│    ├── memory.py              # Modified implementation of the SB3 replay buffer (agent's memory)
│    ├── networks.py            # Definitions of the agent's neural networks
│    ├── train.py               # The training script
│    ├── utils.py               # Various utility methods
│    └── wrappers.py            # Various reward wrappers for the environment
├── misc
│    └── visualize_obs.py       # Script used to create figures for the thesis
├── README.md
└── requirements.txt
```

## Usage Instructions

We used `Python 3.10`.
To install dependencies, run the following command:
```sh
pip install -r requirements.txt
```

### Training

To train an agent, simply run the training script:
```sh
python agent/train.py
```

Additionally you might want to run training with different parameters:
| Option | Description |
| ------ | ----------- |
| `-h, --help` | show this help message and exit |
| `--exp-name` **STR** | the name of this experiment **`(default: '')`** |
| `--seed` **INT** | seed of the experiment **`(default: 1)`** |
| `--torch-deterministic, --no-torch-deterministic` | if toggled, `torch.backends.cudnn.deterministic=False` **`(default: True)`** |
| `--track, --no-track` | if toggled, this experiment will be tracked with Weights and Biases **`(default: False)`** |
| `--wandb-project-name` **STR** | the wandb's project name **`(default: SAC+extensions)`** |
| `--wandb-entity` **STR** | the entity (team) of wandb's project **`(default: '')`** |
| `--wandb-group` **STR** | the group in the wandb project **`(default: '')`** |
| `--capture-video, --no-capture-video` | whether to capture videos of the agent performances (check out `videos` folder) **`(default: False)`** |
| `--evaluate`, `--no-evaluate` | whether to evaluate the agent's deterministic policy during training **`(default: False)`** |
| `--eval_frequency` **INT** | the frequency of evaluation runs **`(default: 50,000)`** |
| `--eval_runs` **INT** | the number of evaluation runs to perform **`(default: 3)`** |
| `--eval_episodes` **INT** | the number of episodes in each evaluation run **`(default: 10)`** |
| `--backend` **STR** | which backend to use: `mps` or `cuda` **`(default: cuda)`** |
| `--save-model, --no-save-model` | whether to save the model at `./models/` **`(default: True)`** |
| **Algorithm specific arguments** | |
| `--env-id` **STR** | the id of the environment **`(default: SpaceInvadersNoFrameskip-v4)`** |
| `--total-timesteps` **INT** | total timesteps of the experiments **`(default: 5,000,000)`** |
| `--buffer-size` **INT** | the replay memory buffer size **`(default: 500,000)`** |
| `--gamma` **FLOAT** | the discount factor gamma **`(default: 0.99)`** |
| `--tau` **FLOAT** | target smoothing coefficient **`(default: 1.0)`** |
| `--batch-size` **INT** | the batch size of sample from the replay memory **`(default: 64)`** |
| `--learning-starts` **INT** | timestep to start learning **`(default: 20,000)`** |
| `--policy-lr` **FLOAT** | the learning rate of the policy network optimizer **`(default: 0.0003)`** |
| `--q-lr` **FLOAT** | the learning rate of the Q network network optimizer **`(default: 0.0003)`** |
| `--update-frequency` **INT** | the frequency of training updates **`(default: 4)`** |
| `--target-network-frequency` **INT** | the frequency of updates for the target networks **`(default: 8,000)`** |
| `--alpha` **FLOAT** | entropy regularization coefficient. **`(default: 0.05)`** |
| `--autotune, --no-autotune` | automatic tuning of the entropy coefficient **`(default: True)`** |
| `--target-entropy-scale` **FLOAT** | coefficient for scaling the autotune entropy target **`(default: 0.89)`** |
| **SD-SAC specific arguments** | |
| `--entropy-penalty, --no-entropy-penalty` | use entropy penalty **`(default: False)`** |
| `--beta` **FLOAT** | entropy-penalty coefficient **`(default: 0.5)`** |
| `--double-avg-w-q-clip, --no-double-avg-w-q-clip` | use double average Q-learning with Q-clip **`(default: False)`** |
| `--c` **FLOAT** | Q-clip range **`(default: 0.5)`** |
| **Munchausen RL specific arguments** | |
| `--munchausen`, `--no-munchausen` | whether to use Munchausen RL **`(default: False)`** |
| `--reward-scale` **FLOAT** | munchausen reward scaling factor **`(default: 0.9)`** |
| `--l0` **FLOAT** | log-policy clip range **`(default: -1)`** |
| **SEER specific arguments** | |
| `--freeze-encoders`, `--no-freeze-encoders` | whether to freeze encoders (SEER) **`(default: False)`** |
| `--store_embeddings`, `--no-store_embeddings` | whether to store embeddings (SEER) **`(default: False)`** |
| `--tf` **INT** | network updates to activate SEER **`(default: 200,000)`** |
| **Environment specific arguments** | |
| `--episodic-life, --no-episodic-life` | whether to use the `EpisodicLifeEnv` wrapper (treat each life as single episode) **`(default: True)`** |
| `--life-loss-penalty, --no-life-loss-penalty` | whether to penalize the agent when it loses a life **`(default: False)`** |
| `--reward-wrapper` **STR** | which reward wrapper to use: `clipped`, `score` or `inverse` **`(default: 'clipped')`** |
| `--num-envs` **INT** | the number of parallel environments running **`(default: 4)`** |

### Evaluation

To evaluate a trained agent, run the following command:
```sh
python agent/eval.py --path {path to models}
```
with the following parameters:
| Option | Description |
| ------ | ----------- |
| `-h, --help` | show this help message and exit |
| `--path` **INT** | the path to the directory of the saved models **`(required)`** |
| `--runs` **INT** | the number of evaluation runs **`(default: 5)`** |
| `--episodes` **STR** | the number of episodes in an evaluation run **`(default: 50)`** |
| `--num-envs` **INT** | the number of parallel environments running **`(default: 8)`** |
| `--backend` **STR** | which backend to use: `mps` or `cuda` **`(default: 'cpu')`** |

## References

- Richard S Sutton. Reinforcement learning: An introduction. A Bradford Book, 2018.
- Kevin Murphy. Reinforcement learning: An overview. arXiv preprint arXiv:2412.05265, 2024.
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.
- Christian L Martinez-Nieves. Defeating the invaders with deep reinforcement learning.
- David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In International conference on machine learning, pages 387–395. Pmlr, 2014.
- Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.
- Hado Hasselt. Double q-learning. Advances in neural information processing systems, 23, 2010.
- Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
- Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas. Dueling network architectures for deep reinforcement learning. In International conference on machine learning, pages 1995–2003. PMLR, 2016.
- Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.
- Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, et al. Noisy networks for exploration. arXiv preprint arXiv:1706.10295, 2017.
- Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In International conference on machine learning, pages 449–458. PMLR, 2017.
- Tyler Clark, Mark Towers, Christine Evers, and Jonathon Hare. Beyond the rainbow: High performance deep reinforcement learning on a desktop pc. arXiv preprint arXiv:2411.03820, 2024.
- Petros Christodoulou. Soft actor-critic for discrete action settings. arXiv preprint arXiv:1910.07207, 2019.
- Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. Pmlr, 2018.
- Haibin Zhou, Zichuan Lin, Junyou Li, Qiang Fu, Wei Yang, and Deheng Ye. Revisiting discrete soft actor-critic. arXiv preprint arXiv:2209.10081, 2022.
- Lili Chen, Kimin Lee, Aravind Srinivas, and Pieter Abbeel. Improving computational efficiency in visual reinforcement learning via stored embeddings. Advances in Neural Information Processing Systems, 34:26779–26791, 2021.
- Nino Vieillard, Olivier Pietquin, and Matthieu Geist. Munchausen reinforcement learning. Advances in Neural Information Processing Systems, 33:4235–4246, 2020.
- Denis Yarats, Amy Zhang, Ilya Kostrikov, Brandon Amos, Joelle Pineau, and Rob Fergus. Improving sample efficiency in model-free reinforcement learning from images. In Proceedings of the aaai conference on artificial intelligence, volume 35, pages 10674–10681, 2021.
- The Centre for Computing History. Space invaders. https://www.computinghistory.org.uk/det/329/Space-Invaders/. Webpage describing the Atari 2600 game Space Invaders. Reference ID CH329. Accessed: 2025-04-14.
- Atari, Inc. Space Invaders Instruction Manual, 1980. Available at: https://atariage.com/manual_html_page.php?SoftwareLabelID=460. Accessed: 2025-04-14.
- M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253–279, jun 2013.
- Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.
- Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.
- Mathieu Poliquin. Stable retro, a maintained fork of openai’s gym-retro. https://github.com/Farama-Foundation/stable-retro, 2025.
- Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius Zambaldi, Satyaki Upadhyay, Julien Pérolat, Sriram Srinivasan, Finbarr Timbers, Karl Tuyls, Shayegan Omidshafiei, et al. Openspiel: A framework for reinforcement learning in games. arXiv preprint arXiv:1908.09453, 2019.
- Jun Jet Tai, Mark Towers, and Elliot Tower. Shimmy: Gymnasium and PettingZoo Wrappers for Commonly Used Environments, June 2023.
- Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021.
- Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jeff Braga, Dipam Chakraborty, Kinal Mehta, and João G.M. Araújo. Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms. Journal of Machine Learning Research, 23(274):1–18, 2022.
- Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. MLX: Efficient and flexible machine learning on apple silicon, 2023.
- Tristan Billot. How fast is mlx? a comprehensive benchmark on 8 apple silicon chips and 4 cuda gpus. Towards Data Science, February 2024. Available at: https://towardsdatascience.com/how-fast-is-mlx-a-comprehensive-benchmark-on-8-apple-silicon-chips-and-4-cuda-gpus-378a0ae356a0/.Accessed: 2025-04-09.
