from rl_modules.environment import Environment
from .DQN_IQL import DQN_model

import torch
from itertools import count
from utils import plotter, logger
from matplotlib import pyplot as plt 


class Trainer:
    def __init__(self, 
                 n_episode,
                 device, 
                 coordinate_seed = 1556) -> None:
        
        self.env = Environment(coordinate_seed = coordinate_seed)
        
        self.device = device
        self.agents = self.env.agents
        self.model = {k : DQN_model(n_actions=self.env.n_actions.get(k), 
                                    n_observations=self.env.n_observations.get(k),
                                    device=self.device) 
                      for k in self.agents}
        self.n_episode = n_episode
    
    def select_action(self, state):
        actions = {}
        for agent in self.agents:
            agent_model = self.model.get(agent)
            agent_state = state.get(agent)
            agent_action = agent_model.select_action(agent_state)
            actions[agent] = agent_action
        
        return actions
    
    def train(self):
        reward_evolution = []
        episode_lifetimes = []
        
        for i in range(self.n_episode):
            logger.debug('TRAIN', f'Episode: {i}') 
            state = self.env.reset()
            state = {k: torch.tensor([state.get(k)],
                                                  device = self.device,
                                                  dtype = torch.float) 
                                  for k in self.agents}
            reward_sum = 0
            for t in count():
                logger.debug('TRAIN', f'Round: {t}') 
                
                actions = self.select_action(state)
                observations, reward, terminal = self.env.step({k : actions.get(k).item() 
                                                                   for k in self.agents})

                reward_sum += reward
                reward = torch.tensor([reward], device=self.device)
                
                if terminal:
                    next_state = {k: None 
                                  for k in self.agents}
                else:
                    next_state = {k: torch.tensor([observations.get(k)],
                                                  device = self.device,
                                                  dtype = torch.float) 
                                  for k in self.agents}
                    
                for agent in self.agents:
                    agent_model = self.model.get(agent)
                    action = torch.tensor([[actions.get(agent)]], device=self.device)
                    agent_model.memory.push(state.get(agent), action, next_state.get(agent), reward)
                    
                    agent_model.optimize_model()
                    agent_model.soft_update()
                
                if terminal:
                    episode_lifetimes.append(t)
                    reward_evolution.append(reward_sum)
                    plotter.plot_durations(episode_lifetimes, reward_evolution)
                    break
        
        plotter.plot_durations(episode_lifetimes, reward_evolution, show_result=True)
        
    

                    
                    
                
            
            
           
                
            
    
    