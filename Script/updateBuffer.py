from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Replay Buffer Class
class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    
    """

    # Init function
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.

        PARAMETERS
        -------------------------
            - action_size (int):  Dimension of each action
            - buffer_size (int):  Maximum size of buffer
            - batch_size (int):   Size of each training batch
            - seed (int):         Random seed
        """
        # The action size object
        self.action_size = action_size
        
        # The memory object
        self.memory = deque(maxlen = buffer_size)  
        
        # The batch size object
        self.batch_size = batch_size
        
        # The exprience object
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        # The seed object
        self.seed = random.seed(seed)
    
    
    # Add function
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        
        PARAMETERS
        -------------------------
            - state
            - action
            - reward
            - next_state
            - done
        """
        # Get the exprience
        e = self.experience(state, action, reward, next_state, done)
        
        # Append the exprience into memory
        self.memory.append(e)
    
    
    # Sample function
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        
        """
        # Get a random batch of exprience
        experiences = random.sample(self.memory, k = self.batch_size)
    
        # Get the state in proper type
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        
        # Get the action in proper type
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        
        # Get the reward in proper type
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        
        # Get the next state in proper type
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        
        # Get the done in proper type
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        # Return the (S,A,R,S',done)
        return (states, actions, rewards, next_states, dones)


    # Function for getting the memory size
    def __len__(self):
        """
        Return the current size of internal memory.
        
        """
        # Length of memory
        return len(self.memory)
