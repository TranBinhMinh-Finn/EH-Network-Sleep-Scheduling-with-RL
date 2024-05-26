from common.consumption_model import send_receive_multiple_packets, MSG_TYPE
from utils.common import get_neighbors, D, calculate_coverage, calculate_coverage_for_each_radius
from enum import Enum
from common.settings import PARAMS
from .EC_CKN import EC_CKN, STATE
import math

radio_range = PARAMS.get('rr')
    
EH_E0 = PARAMS.get('eh_initial_energy')

E_Critical = 2.5 * 100 * 10 ** 9
class EH_EC_CKN(EC_CKN):
    
    eh_nodes = []
    
    extend_eh_nodes_range = False
    
    def __init__(self, manager, eh_nodes) -> None:
        super().__init__(manager)
        self.eh_nodes = eh_nodes
    
    def reset(self):
        # super().reset()  
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

    def compute_sleep_condition(self, node):
        """
        Compute whether the node can sleep or not in the next round.
        """
        if node not in self.eh_nodes:
            return super().compute_sleep_condition(node)
        if node.energy < E_Critical:
            return STATE.SLEEP
        return STATE.AWAKE
    
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
        
        # print(f"k = {self.k}, Coverage: {coverage_sum}")
        
        return coverage_sum
    
    