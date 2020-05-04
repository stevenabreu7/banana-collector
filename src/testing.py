from .agent import Agent
from unityagents import UnityEnvironment


def test_agent(env, brain_name, network_weights, network_layers):
    """Test the given agent in the environment.
        
    Params:
        network_weights (string): filename where network weights are stored

    Returns:
        score (int): score for this episode
    """
    # load the agent
    env_info = env.reset(train_mode=False)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = env.brains[brain_name].vector_action_space_size
    agent = Agent(state_size, action_size, seed=0, network_layers=network_layers)
    agent.load_weights(network_weights)
    # reset environment
    state = env_info.vector_observations[0]
    score = 0
    done = False
    # play one episode
    while not done:
        action = agent.act(state, eps=0.0) # greedy
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
    # return score
    return score