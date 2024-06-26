from common.consumption_model import send_receive_multiple_packets, MSG_TYPE
from utils.common import get_neighbors, D, calculate_coverage
from enum import Enum
from common.settings import PARAMS
from .EC_CKN import EC_CKN, STATE
import torch
from rl_modules.DQN_IQL import DQN_model
import utils.logger as logger

radio_range = PARAMS.get('rr')
    
EH_E0 = PARAMS.get('eh_initial_energy')

class RL_EH_EC_CKN(EC_CKN):
    
    range_extend_value = {}
    model = {}
    last_state = {}
    
    def __init__(self, manager, eh_nodes):
        super().__init__(manager)
        self.eh_nodes = eh_nodes
    
    def reset(self):
        pass
            
    def get_neighbors(self, node, all_nodes):
        neighbors = []
        range_extension = 0
        if node in self.eh_nodes:
            range_extension = self.range_extend_value.get(node)    
                
        for node_ in all_nodes:
            # eh nodes choose their own sleep 
            if node_ in self.eh_nodes and self.state.get(node_) == STATE.SLEEP:
                continue
            if D(node, node_) <= radio_range + range_extension:
                neighbors.append(node_)
        
        return neighbors
    
    def perform_eh_nodes_actions(self, actions):
        for node, action in actions.items():
            if action == 61:
                self.range_extend_value[node] = 0
                self.state[node] = STATE.SLEEP
                # logger.debug('TRAIN', f'Node {node.id} is sleeping.')
            else:
                self.state[node] = STATE.AWAKE
                self.range_extend_value[node] = action
                # logger.debug('TRAIN', f'Node {node.id} is extending range by {action}.')
            
    def get_node_state(self, node, harvested_energy):
        neighbors = self.neighbor_1hop.get(node)
        neighbors_2hop = self.neighbor_2hop.get(node)
        state = [node.energy, harvested_energy]
        for node_ in neighbors:
            if node_.is_dead:
                state.append(0)
            else:
                state.append(node_.energy)
        for node_ in neighbors_2hop:
            if node_.is_dead:
                state.append(0)
            else:
                state.append(node_.energy)
        
        return state        

    def compute_sleep_condition(self, node):
        """
        Compute whether the node can sleep or not in the next round.
        """
        neighbors = self.neighbor_1hop.get(node)
            
        # Stay awake if the node or its neighbors have less than k neighbors 
        
        if len(neighbors) < self.k:
            return STATE.AWAKE
        
        for n in neighbors:
            if len(self.neighbor_1hop.get(n)) < self.k:
                return STATE.AWAKE
        
        E_u = [n for n in neighbors if n.energy > node.energy and (n not in self.eh_nodes or self.state.get(n) == STATE.AWAKE)]
        
        # condition 1: any node in neighbors has k neighbors in e_u
        for n in neighbors:
            neighbor_count = 0
            for v in self.neighbor_1hop.get(n):
                if v in E_u and v is not n:
                    neighbor_count += 1
            if neighbor_count < self.k:
                return STATE.AWAKE
        
        # condition 2: any 2 nodes in e_u is connected directly, or indirectly through a node with erank > erankU 
        for u in E_u:
            for v in E_u:
                if u == v:
                    continue
                if u in self.neighbor_1hop.get(v) or v in self.neighbor_1hop.get(u):
                    continue
                common_neighbors = list(set(self.neighbor_1hop.get(v)).intersection(self.neighbor_1hop.get(u)))
                condition = False
                for n in common_neighbors:
                    if n.energy > node.energy and (n not in self.eh_nodes or self.state.get(n) == STATE.AWAKE):
                        condition = True

                if condition is False:
                    return STATE.AWAKE
        
        return STATE.SLEEP
    
    def perform_scheduling(self, get_neighbors_func = None):
        """
        Execute the EC_CKN algorithm.
        """
        if get_neighbors_func is None:
            get_neighbors_func = self.manager.get_neighbors
            
        for node in self.manager.all_nodes:
            self.broadcast_rank(node, get_neighbors_func)
            
        self.broadcast_rank(self.manager.source_node, get_neighbors_func)
        self.broadcast_rank(self.manager.sink_node, get_neighbors_func)
        
        for node in self.manager.all_nodes:
            self.broadcast_neighbor_ranks(node, get_neighbors_func)
            
        self.broadcast_neighbor_ranks(self.manager.source_node, get_neighbors_func)
        self.broadcast_neighbor_ranks(self.manager.sink_node, get_neighbors_func)
            
        for node in self.manager.all_nodes:
            if node.is_dead or node in self.eh_nodes:
                continue
            self.state[node] = self.compute_sleep_condition(node)
    
    def get_coverage_degree(self, network_size, all_nodes):
        
        x, y = network_size
        
        coverage_sum = 0
        
        for node in all_nodes:
            if self.state.get(node) == STATE.AWAKE:
                range_extension = self.range_extend_value.get(node, 0)
                coverage_sum += calculate_coverage(node, network_size, radio_range + range_extension)
        coverage_sum /= (x * y)
        
        return coverage_sum