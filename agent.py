import copy
import re
import numpy as np
import random

from collections import deque, namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from model import Actor, Critic
from buffers import PrioritizedReplayBuffer, ReplayBuffer

## Agent Global Parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TD3Agent:

    def __init__(self, state_size, action_size, a_max, seed=0,
                 gamma=0.99, tau=0.01, lr=1e-3,
                 noise_clip=0.5, policy_noise=0.2, policy_freq=8, learn_freq=4):
        '''
        Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            a_max (float): maximum action value
            seed (int): random seed
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate
            noise_clip (float): noise clip
            policy_noise (float): policy noise
            policy_freq (int): policy frequency
        '''
        
        self.state_size = state_size
        self.action_size = action_size

        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = random.seed(seed)

        self.a_max = a_max
        self.gamma = gamma  # discount factor
        self.tau = tau      # for soft update of target parameters
        self.lr = lr        # learning rate
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.learn_freq = learn_freq

        self.t_step = 0

        # Memory
        self.memory = PrioritizedReplayBuffer(seed=seed)
        # NOTE: Prioritized Experience Replay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, a_max, seed=seed).to(device)
        self.actor_target = copy.deepcopy(self.actor_local)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-3)
        # NOTE: learning rate scheduler

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed=seed).to(device)
        self.critic_target = copy.deepcopy(self.critic_local)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3)
        # NOTE: learning rate scheduler


        # Noise process TBD

    def step(self, state, action, reward, next_state, done):
        '''
        Save experience in replay memory, and use random sample from buffer to learn.
        '''
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step += 1
        if len(self.memory) > self.memory.batch_size and (self.t_step % self.learn_freq) == 0:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)
    
    def act(self, state, add_noise=True):
        '''
        Returns actions for given state as per current policy.
        '''
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()
        return action
        raise NotImplementedError
    
    def learn(self, experiences, gamma):
        '''
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        # unpack experiences
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            noise = (torch.randn_like(actions.float()) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.a_max, self.a_max)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_local(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # wait for policy_freq steps
        if self.t_step % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_local.Q1(states, self.actor_local(states)).mean()

            # optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        '''
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        '''

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)        
        