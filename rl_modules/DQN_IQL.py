from .replay_memory import ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from utils import logger
class DQN(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DRQN(nn.Module):
    def __init__(self, n_observations, n_actions, device):
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv1d(n_observations, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.gru = nn.GRU(128, 128)
        self.hidden_layer = torch.zeros(1, 128, device=device)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)
        
    def forward(self, x):
        x = x.permute(1, 0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(1, 0)
        x, _ = self.gru(x)
        x = F.relu(self.layer1(F.relu(x)))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.001
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-6

class DQN_model():
    
    def __init__(self, 
                 device,
                 n_observations = 400,
                 n_actions = 62,
                 batch_size = BATCH_SIZE,
                 gamma = GAMMA,
                 eps_start = EPS_START,
                 eps_end = EPS_END,
                 eps_decay = EPS_DECAY,
                 tau = TAU,
                 lr = LR,
                 decay_threshold = 5000,
                 ) -> None:
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.device = device
        self.decay_threshold = decay_threshold
        self.decay_factor = 0
        self.n_actions = n_actions
        self.n_observations = n_observations
        
        self.policy_net = DRQN(self.n_observations, self.n_actions, device).to(device)
        self.target_net = DRQN(self.n_observations, self.n_actions, device).to(device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.sum_loss = 0
        self.count_loss = 0
        self.memory = ReplayMemory(10000)
        
    steps_done = 0   
    
    def select_action(self, state):
        """
        Return the action for a node using epsilon greedy strategy 
        """
        sample = random.random()
        
        if self.steps_done > self.decay_threshold:
            self.decay_factor = self.steps_done - self.decay_threshold
        eps_threshhold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.decay_factor / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshhold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1,1)
        else:
            # pick a random action
            return torch.tensor([[random.randint(0,self.n_actions-1)]], device=self.device)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                                batch.next_state)), 
                                            device = self.device, 
                                            dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
    
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.sum_loss += loss.item()
        self.count_loss += 1

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        if self.steps_done > self.decay_threshold:
            self.lr_scheduler.step()
        
        return 0 if self.count_loss == 0 else self.sum_loss / self.count_loss 
    
    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        
    def loss(self):
        return 0 if self.count_loss == 0 else self.sum_loss / self.count_loss