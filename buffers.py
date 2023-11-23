import numpy as np
import torch

import random
from collections import deque, namedtuple


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size

class ReplayBuffer:
    '''
    Fixed-size buffer to store experience tuples.
    '''

    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0):
        '''
        Initialize ReplayBuffer object
        '''
        self.max_size = buffer_size
        self.batch_size = batch_size
        # namedtuple: factory function for creating tuple subclasses with named fields
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # deque: list-like container with fast appends and pops on either end
        self.memory = deque(maxlen=self.max_size)
        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory.
        '''
        # create experience tuple
        e = self.experience(state, action, reward, next_state, done)
        # append to memory
        self.memory.append(e)
    
    def sample(self):
        '''
        Randomly sample a batch of experiences from memory.
        '''
        # sample experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)
        # generate tensors from experiences 
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        # return tensors
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        '''
        Return the current size of internal memory.
        '''
        return len(self.memory)
    

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.alpha = .6
        self.beta = .4
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(1.)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)
        priorities = torch.tensor(self.priorities, device=device)
        probabilities = (priorities ** self.alpha) / (priorities ** self.alpha).sum()
        
        #self.alpha *= ALPHA_DECAY

        #print(len(self.memory), len(self.priorities), len(probabilities), probabilities.sum())

        indices = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities.detach().cpu().numpy())  
        #indices = torch.multinomial(probabilities, self.batch_size)#, replacement=True)
        experiences = [self.memory[idx] for idx in indices]

        # importance sampling weigth
        weights = (len(self.memory) * probabilities[indices].cpu().numpy()) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=device, dtype=torch.float32).reshape(-1, 1)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
    
    
    def update_priorities(self, indices, errors, offset=1e-5):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + offset


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
