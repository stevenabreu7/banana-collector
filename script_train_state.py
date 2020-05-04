from unityagents import UnityEnvironment
from src.agent import Agent
from src.training import train_agent, show_scores_plot
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

print("\nstart training dqn agent ([512,512], tau=1-3, eps_decay=0.999)\n")
title = "ddqn_512x512_epsd997"
agent = Agent(state_size, action_size, seed=0, double_ql=False, tau=1e-3, network_layers=[512,512])
scores = train_agent(env, brain_name, agent, epsilon_decay=0.999, epsilon_min=0.01, prefix=title + "_")
show_scores_plot(scores, filename="assets/scores_state_{}.png".format(title))

print("\nstart training ddqn agent ([512,512,128], tau=2e-3, eps_decay=0.998)\n")
title = "ddqn_512x256x128"
agent = Agent(state_size, action_size, seed=0, double_ql=True, tau=2e-3, network_layers=[512,512,128])
scores = train_agent(env, brain_name, agent, epsilon_decay=0.998, epsilon_min=0.01, prefix=title + "_")
show_scores_plot(scores, filename="assets/scores_state_{}.png".format(title))

print("\nstart training ddqn agent ([512,512], tau=2e-3, eps_decay=0.997)\n")
title = "ddqn_512x512_epsd997"
agent = Agent(state_size, action_size, seed=0, double_ql=True, tau=2e-3, network_layers=[512,512])
scores = train_agent(env, brain_name, agent, epsilon_decay=0.997, epsilon_min=0.01, prefix=title + "_")
show_scores_plot(scores, filename="assets/scores_state_{}.png".format(title))

print("\nstart training ddqn agent ([256,256,128], tau=2e-3, eps_decay=0.998)\n")
title = "ddqn_256_256_128"
agent = Agent(state_size, action_size, seed=0, double_ql=True, tau=2e-3, network_layers=[256,256,128])
scores = train_agent(env, brain_name, agent, epsilon_decay=0.998, epsilon_min=0.01, prefix=title + "_")
show_scores_plot(scores, filename="assets/scores_state_{}.png".format(title))

print("\nstart training ddqn agent ([512,256,128], tau=2e-3, eps_decay=0.998)\n")
title = "ddqn_512x256x128"
agent = Agent(state_size, action_size, seed=0, double_ql=True, tau=2e-3, network_layers=[512,256,128])
scores = train_agent(env, brain_name, agent, epsilon_decay=0.998, epsilon_min=0.01, prefix=title + "_")
show_scores_plot(scores, filename="assets/scores_state_{}.png".format(title))
