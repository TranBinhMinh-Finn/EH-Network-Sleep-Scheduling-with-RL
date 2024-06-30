from common.simulator import SimManager
from common.consumption_model import send_receive_packets, MSG_TYPE, EnergyHarvesting
from common.settings import PARAMS
from common.routing_protocol import TPGFRouter

from sleep_scheduler.RL_EH_EC_CKN import RL_EH_EC_CKN

import utils.logger as logger

from datetime import datetime

EH_E0 = PARAMS.get('eh_initial_energy')

radio_range = PARAMS.get('rr')
class Environment():
    def __init__(self, 
                 coordinate_seed, 
                 n_actions = radio_range + 2,
                 time_offset = 10 * 60):
        # simulation

        self.manager = SimManager()
        self.manager.make_sink()
        self.manager.make_source()
        self.manager.generate_nodes(seed = coordinate_seed)
        
        self.time_offset = time_offset
        self.eh_model = EnergyHarvesting(time_offset=self.time_offset)
        self.eh_model.pick_eh_nodes(ratio = PARAMS.get('eh_ratio'), 
                                    all_nodes = self.manager.all_nodes)
        
        self.router = TPGFRouter(self.manager)
        
        self.sleep_scheduler = RL_EH_EC_CKN(manager = self.manager, 
                                            eh_nodes = self.eh_model.eh_nodes)
        
        self.D_Target  = PARAMS.get('D_Target')
        self.network_size = PARAMS.get('sensor_network_size')
        
        self.agents = self.eh_model.eh_nodes
        
        self.n_actions = {k : n_actions for k in self.agents}
        self.n_observations = {k : 
            2 + len(self.sleep_scheduler.neighbor_1hop.get(k)) + len(self.sleep_scheduler.neighbor_2hop.get(k)) 
            for k in self.agents}
        
    def reset(self):
        self.manager.reset_nodes()
        for node in self.eh_model.eh_nodes:
            node.energy = EH_E0
        self.eh_model.current_time = self.time_offset
        return {k: self.sleep_scheduler.get_node_state(k, self.eh_model.harvested_energy) for k in self.agents}
    
    def perform_routing_and_transmission(self):
        self.router.reset()
        self.router.choose_nexthop(base_station = self.manager.sink_node, 
                            node = self.manager.source_node, 
                            get_neighbors_func = self.sleep_scheduler.get_awake_neighbors)
        
        current_node = self.manager.source_node
        while current_node != self.manager.sink_node:
            next_hop = self.router.next_hop.get(current_node)
            if next_hop is None:
                logger.debug(f"No path from source node to sink. Stopping episode")
                return False
                
            send_receive_packets(current_node, next_hop, MSG_TYPE.DATA)
            
            current_node = next_hop
        
        return True
    
    def calculate_reward(self, D):
        if D >= self.D_Target:
            return 1
        return -1
    
    def step(self, action):
        self.eh_model.tick()
        self.eh_model.save_energy_states()
        
        self.sleep_scheduler.perform_eh_nodes_actions(action)
        
        self.sleep_scheduler.k = 1
        self.sleep_scheduler.reset()
        self.sleep_scheduler.perform_scheduling(self.sleep_scheduler.get_neighbors)
        
        D = self.sleep_scheduler.get_coverage_degree(network_size = self.network_size, all_nodes = self.manager.all_nodes)
        
        while D < self.D_Target and self.sleep_scheduler.k < self.sleep_scheduler.max_k:
            self.sleep_scheduler.k += 1
            self.sleep_scheduler.perform_scheduling(self.sleep_scheduler.get_neighbors)
            D = self.sleep_scheduler.get_coverage_degree(network_size = self.network_size, all_nodes = self.manager.all_nodes)
            if D >= self.D_Target:
                break
            
        transmission_success = self.perform_routing_and_transmission()

        terminal = False
        if not transmission_success:
            terminal = True
        
        for node in self.manager.all_nodes:
            if node.is_dead and node not in self.eh_model.eh_nodes:
                terminal = True
                
        reward = self.calculate_reward(D)
        
        observations = {k: self.sleep_scheduler.get_node_state(k, self.eh_model.harvested_energy) for k in self.agents}
        
        return observations, reward, terminal, D
    
    filename = f"res/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    
    def get_node_energy_state(self):
        return {node : node.energy for node in self.manager.all_nodes}
    
    def get_eh_node_energy_state(self):
        return {node.id : node.energy for node in self.eh_model.eh_nodes}
    
    def get_non_eh_node_energy_state(self):
        nodes = [node for node in self.manager.all_nodes if node not in self.eh_model.eh_nodes]
        return {node.id : node.energy for node in self.eh_model.eh_nodes}
    
            
                
        
         
    
    