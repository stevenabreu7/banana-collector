# Unity banana collector

Training agents using deep reinforcement learning for the banana collector environment from the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). 

![banana collector environment](./assets/banana.gif)

## Learning task

Goal: 
- collect as many yellow bananas as possible
- +13 average score over 100 episodes

State space: 
- `Box(1)` dimension of agent's velocity
- `Box(36)` dimensions of local ray-cast perception on nearby objects

Actions: 
- `Discrete(4)` forward, backward, left, right

Rewards: 
- +1 for yellow banana
- -1 for blue banana

## Installation

This project is using Python 3.6.3, make sure the following packages are installed:

```
pip install numpy matplotlib torch setuptools bleach==1.5.0 unityagents
```

