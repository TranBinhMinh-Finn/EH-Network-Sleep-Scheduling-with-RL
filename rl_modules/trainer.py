from rl_modules.environment import Environment
from .DQN_IQL import DQN_model

import torch
from itertools import count
from utils import plotter, logger

import numpy as np
import math
class Trainer:
    def __init__(self, 
                 n_episode,
                 device, 
                 coordinate_seed = 1556,
                 save_energy_states = False) -> None:
        
        self.env = Environment(coordinate_seed = coordinate_seed)
        
        self.device = device
        self.agents = self.env.agents
        self.model = {k : DQN_model(n_actions=self.env.n_actions.get(k), 
                                    n_observations=self.env.n_observations.get(k),
                                    device=self.device) 
                      for k in self.agents}
        self.n_episode = n_episode
        self.save_energy_states = save_energy_states
        self.saved_states = {}
        self.saved_loss = {}
    
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
        epsilon_t = []
        loss_t = []
        
        for i in range(self.n_episode):
            logger.debug('TRAIN', f'Episode: {i}') 
            state = self.env.reset()
            state = {k: torch.tensor([state.get(k)],
                                                  device = self.device,
                                                  dtype = torch.float) 
                                  for k in self.agents}
            reward_sum = 0
            
            if self.save_energy_states:
                self.saved_states.clear()

            for t in count():
                logger.debug('TRAIN', f'Round: {t}') 
                
                actions = self.select_action(state)
                observations, reward, terminal = self.env.step({k : actions.get(k).item() 
                                                                   for k in self.agents})
                
                if self.save_energy_states:
                    self.save_energy()
                    eh_states, non_eh_states = self.get_saved_states()
                    plotter.plot_round(self.saved_states)
                    
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
                    loss = agent_model.optimize_model()
                    if self.saved_loss.get(agent) == None:
                        self.saved_loss[agent] = []
                    self.saved_loss[agent].append(loss)
                    
                    agent_model.soft_update()
                
                # if len(self.saved_loss.items()) > 0:
                #     plotter.plot_loss(self.saved_loss)
                if terminal:
                    episode_lifetimes.append(t + 1)
                    reward_evolution.append(reward_sum)
                    
                    agent = self.agents[0]
                    agent_model = self.model.get(agent)
                    eps_threshhold = agent_model.eps_end + (agent_model.eps_start - agent_model.eps_end) * math.exp(-1. * agent_model.decay_factor / agent_model.eps_decay)
                    
                    epsilon_t.append(eps_threshhold)
                    plotter.plot_durations(episode_lifetimes, reward_evolution)
                    # plotter.plot_epsilon(epsilon_t)
                    if self.save_energy_states:
                        plotter.plot_round(self.saved_states, show_result=True)
                    # plotter.plot_loss(self.saved_loss)
                    break
        
        plotter.plot_durations(episode_lifetimes, reward_evolution, show_result=True)
        
    def save_energy(self):
        state = self.env.get_node_energy_state()
        for node, energy in state.items():
            if self.saved_states.get(node) == None:
                self.saved_states[node] = []    
            self.saved_states[node].append(energy)


                    
                    
                
            
            
           
                
            
    
    