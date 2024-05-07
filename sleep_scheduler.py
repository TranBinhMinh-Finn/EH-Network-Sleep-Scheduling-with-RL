from consumption_model import send_receive_packets, send_receive_multiple_packets, MSG_TYPE
from utils import get_neighbors, calculate_coverage_for_each_radius, calculate_coverage_radius
from enum import Enum
from settings import PARAMS
import math
import concurrent.futures

class STATE(Enum):
    AWAKE = True
    SLEEP = False

radio_range = PARAMS.get('rr')

# potential class to be both sleep scheduling algo class parent
class SleepScheduler():
    pass

class EC_CKN(): 
    
    energy_rank = {}
    neighbor_energy_rank = {}
    neighbor_2hop_energy_rank = {}
    state = {}
    k = 1
    
    def __init__(self, manager) -> None:
        self.manager = manager
    
    def broadcast_rank(self, node, get_neighbors_func):
        """
        Broadcast the ERank_U of node U to its neighbors. Form the energy rank vector of the node's neighbors N_U.
        """
        ### TODO: redesign this code after finishing perform_scheduling to store data more efficiently
        
        neighbors = get_neighbors_func(node, all_nodes=self.manager.all_nodes)
        send_receive_multiple_packets(node, neighbors, MSG_TYPE.HELLO)
        n_erank = []
        self.neighbor_energy_rank[node] = neighbors
            
    def broadcast_neighbor_ranks(self, node, get_neighbors_func):
        """
        Broadcast energy rank vector N_U of the node U's neighbors. Form the energy rank vector of the node's 2 hop neighbor S_U.
        """
        
        ### TODO: redesign this code after finishing perform_scheduling to store data more efficiently
         
        neighbors = get_neighbors_func(node, all_nodes=self.manager.all_nodes)
        send_receive_multiple_packets(node, neighbors, MSG_TYPE.HELLO)
        s_erank = []
        self.neighbor_2hop_energy_rank[node] = neighbors
    
    def compute_sleep_condition(self, node):
        """
        Compute whether the node can sleep or not in the next round.
        """
        
        neighbors = self.neighbor_energy_rank.get(node)
            
        # Stay awake if the node or its neighbors have less than k neighbors 
        
        if len(neighbors) < self.k :
            return STATE.AWAKE
        
        for n in neighbors:
            if len(self.neighbor_energy_rank.get(n)) < self.k:
                return STATE.AWAKE
        
        E_u = [n for n in neighbors if n.energy > node.energy]
        
        # condition 1: any node in neighbors has k neighbors in e_u
        for n in neighbors:
            neighbor_count = 0
            for v in self.neighbor_energy_rank.get(n):
                if v in E_u and v is not n:
                    neighbor_count += 1
            if neighbor_count < self.k:
                return STATE.AWAKE
        
        # condition 2: any 2 nodes in e_u is connected directly, or indirectly through a node with erank > erankU 
        for u in E_u:
            for v in E_u:
                if u == v:
                    continue
                if u in self.neighbor_energy_rank.get(v) or v in self.neighbor_energy_rank.get(u):
                    continue
                common_neighbors = list(set(self.neighbor_energy_rank.get(v)).intersection(self.neighbor_energy_rank.get(u)))
                condition = False
                for n in common_neighbors:
                    if n.energy > node.energy:
                        condition = True

                if condition is False:
                    return STATE.AWAKE
        
        return STATE.SLEEP
        
    def perform_scheduling(self):
        """
        Execute the EC_CKN algorithm.
        """
        for node in self.manager.all_nodes:
            self.broadcast_rank(node, self.manager.get_neighbors)
            
        self.broadcast_rank(self.manager.source_node, self.manager.get_neighbors)
        self.broadcast_rank(self.manager.sink_node, self.manager.get_neighbors)
        
        for node in self.manager.all_nodes:
            self.broadcast_neighbor_ranks(node, self.manager.get_neighbors)
            
        self.broadcast_neighbor_ranks(self.manager.source_node, self.manager.get_neighbors)
        self.broadcast_neighbor_ranks(self.manager.sink_node, self.manager.get_neighbors)
            
        for node in self.manager.all_nodes:
            if node.is_dead:
                continue
            self.state[node] = self.compute_sleep_condition(node)
        
    def reset(self):
        """
        Reset variables of the algorithm.
        """
        self.energy_rank.clear()
        self.neighbor_energy_rank.clear()
        self.neighbor_2hop_energy_rank.clear()
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
        """
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit each item in the list for processing
            # The executor will schedule the items to run in separate threads
            futures = [executor.submit(calculate_coverage, node, network_size, radio_range) for node in all_nodes]

            # Wait for all the threads to complete
            concurrent.futures.wait(futures)
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                coverage_sum += result
        """
        
        for node in all_nodes:
            if self.state.get(node) == STATE.AWAKE:
                coverage_sum += calculate_coverage(node, network_size, radio_range)
        coverage_sum /= (x * y)
        
        print(f"k = {self.k}, Coverage: {coverage_sum}")
        
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