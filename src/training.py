from .agent import Agent
from collections import deque
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
import sys


def train_agent(env, brain_name, agent, n_episodes=4000, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, prefix=""):
    """Train the given agent in the environment.
        
    Params:
        n_episodes (int): how many episodes to train the agent for
        epsilon (float): starting value for epsilon (epsilon-greedy)
        epsilon_decay (float): decay factor for epsilon (after each episode)
        epsilon_min (float): minimum value for epsilon

    Returns:
        scores (list(int)): score for each episode
    """
    scores = []
    scores_window = deque(maxlen=100)
    solved = False
    for i_episode in range(1, n_episodes+1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get current state
        state = env_info.vector_observations[0]
        # initialize score
        score = 0
        while True:
            # select action
            action = agent.act(state, epsilon)
            # execute action
            env_info = env.step(action)[brain_name]
            # get next_state, reward, done
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            # training step for agent
            agent.step(state, action, reward, next_state, done)
            # update score and state
            score += reward
            state = next_state
            if done:
                break
        # save scores
        scores.append(score)
        scores_window.append(score)
        # update epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        # print scores
        print('\repisode {}\taverage score: {:.2f}'.format(
            i_episode, np.mean(scores_window)
        ), end="\n" if i_episode % 100 == 0 else "")
        sys.stdout.flush()
        # save parameters if environment solved
        if np.mean(scores_window) >= 13.0 and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)
            ))
            solved = True
            agent.save_params(prefix=prefix, suffix="_solv")
    agent.save_params(prefix=prefix, suffix="_end")
    return scores


def test_agent(env, brain_name, agent, n_episodes):
    """Test the given agent in the environment.
        
    Params:
        n_episodes (int): how many episodes to run the agent for

    Returns:
        scores (list(int)): score for each episode
    """
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state) # greedy
            sys.stdout.flush()
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        print('episode {}\tscore: {:.2f}'.format(i_episode, np.mean(score)))
    return scores


def show_scores_plot(scores, filename=None, save_np=True):
    """Show the scores plot and optionally save the plot.
        
    Params:
        filename (str): optional filename to save the plot
    """
    if save_np:
        np.save(filename.split(".")[0] + ".npy" if filename else "scores.npy", scores)
    window_size = 100
    scores = [np.mean(scores[max(0, idx-window_size+1):idx+1]) for idx in range(len(scores))]
    scores[:100] = 0 # first 100 scores are not averaged
    # plot agent scores
    plt.plot(np.arange(1, len(scores)+1), scores, label="agent")
    # plot agent scores that solved environment
    win_scores = scores.copy()
    win_scores[win_scores < 13] = np.nan
    plt.plot(np.arange(1, len(scores)+1), win_scores, label="solved", color="orange")
    # marker for target score (solution)
    plt.hlines(13, 1, len(scores), colors=["red"], linestyles=["dashed"], label="goal")
    # marker for max target episode for finding solution
    plt.vlines(1800, 0, max(max(scores), 13), colors=["red"], linestyles=["dashed"])
    # marker for episode when solution was first found
    ep_solve = np.argmax(np.array(scores) > 13)
    ep_solve = ep_solve
    if ep_solve:
        plt.vlines(ep_solve, 0, max(max(scores), 13), colors=["green"], linestyles=["dashed"], label="solution")
        plt.annotate("{}".format(ep_solve), (ep_solve - len(scores) / 15, 4.0))
    # labels and legend
    plt.ylabel("average score (over 100 episodes)")
    plt.xlabel("episode")
    plt.legend(bbox_to_anchor=(1.17, 1))
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
