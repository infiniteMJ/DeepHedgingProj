import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.module):
    def __init__(self, state_size, action_size, hidden_size1=64, hidden_size2=32):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_value = F.tanh(self.fc3(x))
        return action_value
