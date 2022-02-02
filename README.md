# Reinforcement-Learning
Anatomical landmarks localization using Reinforcement Learning

Formulating landmark localization as a reinforcement learning (RL) problem has typically been using value-based methods. In this project, the actor-critic based policy search method has been used to solve the localization problem. Localization in 3D images yields a large state space and which can result in many trials for finding the optimal policy. To overcome this, the state space is divided into three partial sub domains, one along each axis. RL agents learn optimal policy along each sub domain and then the partial policies are combined into an optimal policy. This ensures faster convergence due to parallel computation along each axis
