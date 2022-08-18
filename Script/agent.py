import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNet:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.qnet_local = QNet(state_size, action_size).to(device)
        self.qnet_target = QNet(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # The replay memory object
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        
        # The time step object (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Step function
    def step(self, state, action, reward, next_state, done):
        """
        Step function.
        
        PARAMETERS
        -------------------------
            - state
            - action
            - reward
            - next_state 
            - done
        """
        # Append experience (S,A,R,S',done) into replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Every UPDATE_EVERY time
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            
            # If length of replay memory gets larger than batch size
            if len(self.memory) > BATCH_SIZE:
                
                # Get a random subset
                experiences = self.memory.sample()
                
                # Learn
                self.learn(experiences, GAMMA)

                
    # Act function
    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        
        PARAMETERS
        -------------------------
            - state (array_like): Current state
            - eps (float):        Epsilon, for epsilon-greedy action selection
        """
        # Get the state in a proper type
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # TODO
        self.qnet_local.eval()
        
        # Disable gradient calculation
        with torch.no_grad():
            
            # TODO
            action_values = self.qnet_local(state)

        # TODO
        self.qnet_local.train()

        ### Epsilon-greedy action selection
        # If random number larger than epsilon
        if random.random() > eps:
            
            # Get the index of the maximum value in action values
            return np.argmax(action_values.cpu().data.numpy())
        
        # If random number less than epsilon
        else:
            
            # Randomely choose in action values
            return random.choice(np.arange(self.action_size))

        
    # Learn function
    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        PARAMETERS
        -------------------------
            - experiences (Tuple[torch.Tensor]):  Tuple of (s, a, r, s', done) tuples 
            - gamma (float):                      Discount factor
        """
        # Get the expriences (S,A,R,S',done)
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnet_local(states).gather(1, actions)

        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnet_local, self.qnet_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        
                        θ_target = τ*θ_local + (1 - τ)*θ_target

        PARAMETERS
        -------------------------
            - local_model (PyTorch model):   Weights will be copied from
            - target_model (PyTorch model):  Weights will be copied to
            - tau (float):                   Interpolation parameter 
        """
        # Iterate through target model & local model
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            
            # Get the soft update formula
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
