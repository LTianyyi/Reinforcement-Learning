# Reinforcement-Learning
Implementation of some reinforcement learning code in TensorFlow 1.14.0.
However, all code for **continuous space**.
```
conda/pip install gym
conda/pip install tensorflow-gpu==1.14.0
```

## Description
1. ddpg_gym: The DDPG(Deep Deterministic Policy Gradient) for gym-Pendulum.
2. td3_gym: The TD3(Twin Delayed Deep Deterministic Policy Gradients) for gym-Pendulum.
  > NOTE: TD3 is not adde with the *Target Policy Smoothing*, maybe added in the future.
3. sac+gym: The SAC(Soft Actor Critic) for gym-Pendulum.

## TODO
1. Modify code comments
2. Add *Target Policy Smoothing* in *td3_gym*.
3. Add *evaluePolicy* in *ddpg_dym* and *td3_gym*.
4. Modularization, separate the modules such as buffer, actor and critic.
