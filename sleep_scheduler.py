from consumption_model import send_receive_packets, send_receive_multiple_packets, MSG_TYPE
from utils import get_neighbors, calculate_coverage_for_each_radius, calculate_coverage_radius, D, calculate_coverage
from enum import Enum
from settings import PARAMS
import math
from rl_model.model import DQN_model
import torch

class STATE(Enum):
    AWAKE = True
    SLEEP = False

radio_range = PARAMS.get('rr')

# potential class to be both sleep scheduling algo class parent
class SleepScheduler():
    pass

class EC_CKN(): 
    
    neighbor_1hop = {}
    neighbor_2hop = {}
    state = {}
    k = 1
    
    def __init__(self, manager) -> None:
        self.manager = manager
        
        for node in manager.all_nodes:
            self.neighbor_1hop[node] = get_neighbors(node, manager.all_nodes)  
        for node in manager.all_nodes:
            self.neighbor_2hop[node] = []
            neighbors = self.neighbor_1hop.get(node)
            for node_ in manager.all_nodes:
                if node_ in neighbors or node == node_:
                    continue
                for neighbor in neighbors:
                    if node_ in self.neighbor_1hop.get(neighbor):
                        self.neighbor_2hop.get(node).append(node_)
                        break
                    
    def broadcast_rank(self, node, get_neighbors_func):
        """
        Broadcast the ERank_U of node U to its neighbors. Form the energy rank vector of the node's neighbors N_U.
        """
        neighbors = get_neighbors_func(node, all_nodes=self.manager.all_nodes)
        send_receive_multiple_packets(node, neighbors, MSG_TYPE.HELLO)
            
    def broadcast_neighbor_ranks(self, node, get_neighbors_func):
        """
        Broadcast energy rank vector N_U of the node U's neighbors. Form the energy rank vector of the node's 2 hop neighbor S_U.
        """ 
        neighbors = get_neighbors_func(node, all_nodes=self.manager.all_nodes)
        send_receive_multiple_packets(node, neighbors, MSG_TYPE.HELLO)
    
    def compute_sleep_condition(self, node):
        """
        Compute whether the node can sleep or not in the next round.
        """
        
        neighbors = self.neighbor_1hop.get(node)
            
        # Stay awake if the node or its neighbors have less than k neighbors 
        
        if len(neighbors) < self.k :
            return STATE.AWAKE
        
        for n in neighbors:
            if len(self.neighbor_1hop.get(n)) < self.k:
                return STATE.AWAKE
        
        E_u = [n for n in neighbors if n.energy > node.energy]
        
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
                    if n.energy > node.energy:
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
            if node.is_dead:
                continue
            self.state[node] = self.compute_sleep_condition(node)
        
    def reset(self):
        """
        Reset variables of the algorithm.
        """
        self.state.clear()
            
    def get_awake_neighbors(self, node, all_nodes):
        """
        Get all awake neighboring nodes according to the scheduling algorithm.
        """
        neighbors = get_neighbors(node, all_nodes)
        awake_neighbors = [n for n in neighbors if self.state.get(n) == STATE.AWAKE]
        return awake_neighbors
    
    def get_coverage_degree(self, network_size, all_nodes):
        
        x, y = network_size
        
        coverage_sum = 0
        
        for node in all_nodes:
            if self.state.get(node) == STATE.AWAKE:
                coverage_sum += calculate_coverage(node, network_size, radio_range)
        coverage_sum /= (x * y)
        
        print(f"k = {self.k}, Coverage: {coverage_sum}")
        
        return coverage_sum

EH_E0 = PARAMS.get('eh_initial_energy')
class EH_EC_CKN(EC_CKN):
    
    eh_nodes = []
    
    extend_eh_nodes_range = False
    
    def __init__(self, manager, eh_nodes) -> None:
        super().__init__(manager)
        self.eh_nodes = eh_nodes
    
    def reset(self):
        super().reset()  
        self.extend_eh_nodes_range = False
        
    def get_neighbors(self, node, all_nodes):
        neighbors = []
        range_extension = 0
        if self.extend_eh_nodes_range and node in self.eh_nodes:
            if node in self.eh_nodes and node.energy * 5 > EH_E0:
                range_extension = math.floor(node.energy / EH_E0 * radio_range)
                
        for node_ in all_nodes:
            if D(node, node_) <= radio_range + range_extension:
                neighbors.append(node_)
        
        return neighbors
    
    def get_awake_neighbors(self, node, all_nodes):
        """
        Get all awake neighboring nodes according to the scheduling algorithm. Account for eh nodes that has their range extended.
        """
        neighbors = self.get_neighbors(node, all_nodes)
        return [n for n in neighbors if self.state.get(n) == STATE.AWAKE]

    def get_coverage_degree(self, network_size, all_nodes):
        
        x, y = network_size
        
        coverage_sum = 0
        
        for node in all_nodes:
            if self.state.get(node) == STATE.AWAKE:
                range_extension = 0
                if self.extend_eh_nodes_range and node in self.eh_nodes:
                    if node in self.eh_nodes and node.energy * 5 > EH_E0:
                        range_extension = math.floor(node.energy / EH_E0 * radio_range)
                coverage_sum += calculate_coverage(node, network_size, radio_range + range_extension)
        coverage_sum /= (x * y)
        
        print(f"k = {self.k}, Coverage: {coverage_sum}")
        
        return coverage_sum

class RL_EH_EC_CKN(EC_CKN):
    
    range_extend_value = {}
    model = {}
    last_state = {}
    
    def __init__(self, manager, eh_nodes) -> None:
        super().__init__(manager)
        self.eh_nodes = eh_nodes
        for node in self.eh_nodes:
            self.model[node] = DQN_model(n_observations= 2 + len(self.neighbor_1hop.get(node)) + len(self.neighbor_2hop.get(node)))

    def get_neighbors(self, node, all_nodes):
        neighbors = []
        range_extension = 0
        if node in self.eh_nodes:
            range_extension = self.range_extend_value.get(node)    
                
        for node_ in all_nodes:
            # eh nodes choose their own sleep 
            if node_ in self.eh_nodes and self.state.get(node_) != STATE.AWAKE:
                continue
            if D(node, node_) <= radio_range + range_extension:
                neighbors.append(node_)
        
        return neighbors
    
    def perform_eh_nodes_sleep_scheduling(self, harvested_energy):
        for node in self.eh_nodes:
            self.last_state[node] = self.get_state(node, harvested_energy)
            action = self.model.get(node).select_action(self.last_state[node])[0].item()
            if action == 61:
                self.range_extend_value[node] = 0
                self.state[node] = STATE.SLEEP
            else:
                self.state[node] = STATE.AWAKE
                self.range_extend_value[node] = action
            
                
    def get_state(self, node, harvested_energy):
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
                
        return torch.tensor([state], device=device, dtype=torch.float)
    
    def update_model(self, harvested_energy, reward):
        """
        Update new experiences for the model
        """
        for node in self.eh_nodes:
            last_state = self.last_state.get(node)
        
            state = self.get_state(node, harvested_energy)
            # get action:
            if self.state.get(node) == STATE.SLEEP:
                action = 61
            else:
                action = self.range_extend_value.get(node)
            
            action = torch.tensor([[action]])
            reward = torch.tensor([[reward]])
            model = self.model.get(node)
            
            model.memory.push(last_state, action, state, reward)
            model.optimize_model()
            model.soft_update()
    
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
        
        # print(f"k = {self.k}, Coverage: {coverage_sum}")
        
        return coverage_sum
            
calculate_coverage_for_each_radius()

def calculate_coverage(node, network_size, rr = radio_range):
    x, y = network_size
    x_node, y_node = node.coordinates    
    coverage_degree = 0
    top_cutoff = rr
    bottom_cutoff = rr
    left_cutoff = rr
    right_cutoff = rr
    if x_node + rr > x:
        right_cutoff = x - x_node 
    if x_node - rr < 0:
        left_cutoff = x_node 
    if y_node + rr > y:
        top_cutoff = y - y_node 
    if y_node - rr < 0:
        bottom_cutoff = y_node
     
    return calculate_coverage_radius(radius=rr, top_cutoff=top_cutoff, bottom_cutoff=bottom_cutoff, left_cutoff=left_cutoff, right_cutoff=right_cutoff)
    """
    for i in range(x):
        for j in range(y):
            if math.sqrt((x_node - (i + 0.5)) ** 2 + (y_node - (j + 0.5)) ** 2) <= rr: 
                coverage_degree += 1
    """
    return coverage_degree