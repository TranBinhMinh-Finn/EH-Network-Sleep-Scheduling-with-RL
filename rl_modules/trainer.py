from rl_modules.environment import Environment
from .DQN_IQL import DQN_model

import torch
from itertools import count
from utils import plotter, logger

import numpy as np
import math
import csv
import os

class Trainer:
    def __init__(self, 
                 n_episode,
                 device, 
                 episode_metrics_save_file,
                 timestep_metrics_save_file,
                 model_save_file,
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
        
        self.episode_metrics_save_file = episode_metrics_save_file
        self.timestep_metrics_save_file = timestep_metrics_save_file
        self.model_save_file = model_save_file
    
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
        
        
        for i in range(self.n_episode):
            n_rounds_satisfy_coverage = 0
            loss_t = []
            
            logger.info('TRAIN', f'Episode: {i}/{self.n_episode}') 
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
                observations, reward, terminal, coverage = self.env.step({k : actions.get(k).item() 
                                                                   for k in self.agents})
                if reward > 0:
                    n_rounds_satisfy_coverage += 1 
                
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
                
                round_loss = []
                for agent in self.agents:
                    agent_model = self.model.get(agent)
                    action = torch.tensor([[actions.get(agent)]], device=self.device)
                    agent_model.memory.push(state.get(agent), action, next_state.get(agent), reward)
                    loss = agent_model.optimize_model()
                    agent_model.soft_update()
                    
                    # if self.saved_loss.get(agent) == None:
                    #     self.saved_loss[agent] = []
                    # self.saved_loss[agent].append(loss)
                    round_loss.append(loss)
                
                if None not in round_loss:
                    loss_t.append(round_loss)
                # if len(self.saved_loss.items()) > 0:
                #     plotter.plot_loss(self.saved_loss)
                
                self.save_timestep(episode=i, 
                                   timestep=t, 
                                   loss=round_loss, 
                                   coverage=coverage)

                if terminal:
                    logger.info('TRAIN', f'Lifetime: {t + 1}')
                    logger.info('TRAIN', f'Reward Sum: {reward_sum}')
                    episode_lifetimes.append(t + 1)
                    reward_evolution.append(reward_sum)
                    agent = self.agents[0]
                    agent_model = self.model.get(agent)
                    eps_threshhold = agent_model.eps_end + (agent_model.eps_start - agent_model.eps_end) * math.exp(-1. * agent_model.decay_factor / agent_model.eps_decay)
                    
                    avg_loss = np.average(loss_t, axis = 0)
                    self.save_episode(episode=i,
                                      reward_sum=reward_sum,
                                      avg_loss=avg_loss,
                                      lifetime=t+1,
                                      n_rounds_with_coverage=n_rounds_satisfy_coverage)
                    
                    plotter.plot_durations(episode_lifetimes, reward_evolution)
                    
                    # save: episode, reward sum, lifetime, rounds_satisfies_coverage, avg_loss
                    
                    # epsilon_t.append(eps_threshhold)
                    # plotter.plot_epsilon(epsilon_t)
                    #if self.save_energy_states:
                    #    plotter.plot_round(self.saved_states, show_result=True)
                    # plotter.plot_loss(self.saved_loss)
                    break
        
        plotter.plot_durations(episode_lifetimes, reward_evolution, show_result=True)
        
    def save_energy(self):
        state = self.env.get_node_energy_state()
        for node, energy in state.items():
            if self.saved_states.get(node) == None:
                self.saved_states[node] = []    
            self.saved_states[node].append(energy)

    def save_episode(self, episode, reward_sum, avg_loss, lifetime, n_rounds_with_coverage):
        episode_csv = self.episode_metrics_save_file
        file_exists = os.path.isfile(episode_csv)
        headers = ['episode', 'lifetime', 'n_rounds_with_coverage', 'reward_sum', 'avg_loss']
        with open(episode_csv, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=headers)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow({
                
                'episode': episode, 
                'reward_sum': reward_sum,
                'avg_loss': avg_loss,
                'lifetime': lifetime,
                'n_rounds_with_coverage': n_rounds_with_coverage,
            })
            
    def save_timestep(self, episode, timestep, loss, coverage):
        timestep_csv = self.timestep_metrics_save_file
        file_exists = os.path.isfile(timestep_csv)
        headers = ['episode', 'timestep', 'loss', 'coverage']
        with open(timestep_csv, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=headers)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow({
                'episode': episode, 
                'timestep': timestep, 
                'loss': loss, 
                'coverage':coverage,
            })
            
    def save_model(self):
        state_dicts = {}
        for agent in self.agents:
            agent_model = self.model.get(agent)
            state_dicts[agent.id] = agent_model.target_net.state_dict()
        torch.save(state_dicts, self.model_save_file)

                    
                    
                
            
            
           
                
            
    
    