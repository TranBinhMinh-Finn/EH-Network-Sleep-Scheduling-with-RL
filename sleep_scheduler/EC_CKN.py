from common.consumption_model import send_receive_multiple_packets, MSG_TYPE
from utils.common import get_neighbors, D, calculate_coverage
from enum import Enum
from common.settings import PARAMS

class STATE(Enum):
    AWAKE = True
    SLEEP = False

radio_range = PARAMS.get('rr')

class EC_CKN(): 
    
    neighbor_1hop = {}
    neighbor_2hop = {}
    state = {}
    k = 1
    
    def __init__(self, manager) -> None:
        self.manager = manager
        self.max_k = 0
        
        for node in manager.all_nodes:
            self.neighbor_1hop[node] = get_neighbors(node, manager.all_nodes)  
            if len(self.neighbor_1hop[node]) > self.max_k:
                self.max_k = len(self.neighbor_1hop[node]) 
                
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