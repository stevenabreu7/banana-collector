from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size=100000, batch_size=64, gamma=0.99, tau=1e-3,\
                lr=5e-4, update_freq=4, double_ql=True, a_prioritization=0.0):
        """Initialize an Agent object.
        
        Params:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            buffer_size (int): size of the replay buffer
            batch_size (int): batch size
            gamma (float): discount factor
            tau (float): interpolation parameter for soft updates
            lr (float): learning rate for neural network optimizer
            update_freq (int): run learning update every ... time steps
            double_ql (boolean): use different networks for estimating and selecting actions when learning
            a_prioritization (float): parameter for prioritization of episodes in memory (default: random)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        # discount factor
        self.gamma = gamma
        # for soft update of target parameters
        self.tau = tau
        # how often to update target network
        self.update_freq = update_freq
        # use double q-learning
        self.double_ql = double_ql
        # prioritization parameter for memory replay
        self.a_prioritization = a_prioritization

        if a_prioritization:
            # TODO: implement prioritization
            print("[WARNING] prioritization is not implemented yet")

        # q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, a_prioritization)
        # initialize time step
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # learn every 'k' time steps
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # if enough samples available, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:
            action (int): action to be executed in this state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # e-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Implements double q-learning.

        Params:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.double_ql:
            # get next predicted actions from target network
            next_actions = self.qnetwork_target(next_states).detach().argmax(dim=1).unsqueeze(1)
            # get max predicted Q values (target network actions) from local network
            Q_targets_next = self.qnetwork_local(next_states).detach().gather(1, next_actions)
        else:
            # get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save_params(self):
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, a_prioritization):
        """Initialize a ReplayBuffer object.

        Params:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a_prioritization (float): parameter for prioritization in queue
                0 - no prioritization, 1 - strict prioritization
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.a_prioritization = a_prioritization
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory.

        Returns:
            experiences (tuple(s, a, r, s', d)): tuple of lists of states, actions, rewards, next states and done
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
