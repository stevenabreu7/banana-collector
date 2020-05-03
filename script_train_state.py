from unityagents import UnityEnvironment
from src.agent import Agent
from src.training import train_agent, test_agent, show_scores_plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['figure.figsize'] = 10, 5

print("\nload environment\n")
env = UnityEnvironment(file_name="src/exec/Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state_size = brain.vector_observation_space_size

print("\nstart training ddqn agent ([512,256,128], tau=2e-3, eps_decay=0.998)\n")
agent = Agent(state_size, action_size, seed=0, double_ql=True, tau=2e-3, network_layers=[512,256,128])
scores = train_agent(env, brain_name, agent, n_episodes=2000, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01)
# scores_test = test_agent(env, brain_name, agent, n_episodes=1)
show_scores_plot(scores, filename="assets/scores_state_ddqn.png")
