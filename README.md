# Unity banana collector

Training agents using deep reinforcement learning for the banana collector environment from the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). There are two versions of the environment, the state-based and the pixel-based environment. They only differ in their coding of the state space (see below).

![banana collector environment](./assets/banana.gif)

## Learning task

Goal: 
- collect as many yellow bananas as possible

State space (state-based):
- `Box(1)` dimension of agent's velocity
- `Box(36)` dimensions of local ray-cast perception on nearby objects

State space (pixel-based):
- `Box(84,84,3)` RGB image of the agent's first person view of the environment

Actions: 
- `Discrete(4)` forward, backward, left, right

Rewards: 
- +1 for yellow banana
- -1 for blue banana

The agent solves the environment when it achieves a +13 average score over 100 episodes.

## Installation

This project is using Python 3.6.3, make sure the following packages are installed:

```bash
pip install numpy matplotlib torch setuptools bleach==1.5.0 unityagents
```

Download the environment executable into `src/exec/`:

- OSX: [state-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip), [pixel-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Linux: [state-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip), [pixel-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Windows (32-bit): [state-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip), [pixel-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [state-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip), [pixel-based](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Make sure to update the `file_name` parameter in the code when loading the environment:

```python
env = UnityEnvironment(file_name="src/exec/...")
```

## Training the agent

All code for training is in [training.py](./src/training.py). You can execute the code as shown in this [notebook](./train_state_based.ipynb) for training an agent for the state-based environment.
