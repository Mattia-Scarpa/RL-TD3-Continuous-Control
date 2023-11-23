"""
Author: Mattia Scapra (mattia.scarpa.1@phd.unipd.it)
"""

import copy
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=[256, 256], seed=0):
        super(Actor, self).__init__()
        
        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.fc = nn.ModuleList()
        # input layer
        self.fc.append(nn.Linear(state_dim, hidden_dim[0]))
        # hidden layers
        for i in range(len(hidden_dim)-1):
            self.fc.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        # output layer
        self.fc.append(nn.Linear(hidden_dim[-1], action_dim))

        # max action
        self.max_action = max_action
        


    def forward(self, state):
        """
        Forward pass
        """
        a = state
        for i in range(len(self.fc)-1):
            a = F.relu(self.fc[i](a))
        a = self.max_action * torch.tanh(self.fc[-1](a))
        return a
    


class Critic(nn.Module):
    """
    Twin critic network
    """

    def __init__(self, state_dim, action_dim, hidden_dim_q1=[256, 256], hidden_dim_q2=[256, 256], seed = 0):
        super(Critic, self).__init__()

        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Q1 architecture
        self.fc1 = nn.ModuleList()
        # input layer
        self.fc1.append(nn.Linear(state_dim + action_dim, hidden_dim_q1[0]))
        # hidden layers
        for i in range(len(hidden_dim_q1)-1):
            self.fc1.append(nn.Linear(hidden_dim_q1[i], hidden_dim_q1[i+1]))
        # output layer
        self.fc1.append(nn.Linear(hidden_dim_q1[-1], 1))

        # Q2 architecture
        self.fc2 = nn.ModuleList()
        # input layer
        self.fc2.append(nn.Linear(state_dim + action_dim, hidden_dim_q2[0]))
        # hidden layers
        for i in range(len(hidden_dim_q2)-1):
            self.fc2.append(nn.Linear(hidden_dim_q2[i], hidden_dim_q2[i+1]))
        # output layer
        self.fc2.append(nn.Linear(hidden_dim_q2[-1], 1))

    def forward(self, state, action):
        """
        Forward pass on both Q1 and Q2
        """
        sa = torch.cat([state, action], 1)
        q1 = sa
        q2 = sa
        for i in range(len(self.fc1)-1):
            q1 = F.relu(self.fc1[i](q1))
            q2 = F.relu(self.fc2[i](q2))
        q1 = self.fc1[-1](q1)
        q2 = self.fc2[-1](q2)
        return q1, q2
    
    def Q1(self, state, action):
        """
        Forward pass on Q1
        """
        sa = torch.cat([state, action], 1)
        q1 = sa
        for i in range(len(self.fc1)-1):
            q1 = F.relu(self.fc1[i](q1))
        q1 = self.fc1[-1](q1)
        return q1