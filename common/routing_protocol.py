from common.settings import PARAMS
from common.consumption_model import D
from utils.common import get_neighbors

radio_range = PARAMS.get('rr')

class TPGFRouter():
    """
    Create the routing path for nodes in a round according to the TPGF Protocol.
    """
    
    def __init__(self, manager) -> None:
        self.manager = manager
        self.nodes_mark = {}
        self.next_hop = {}
    
    def reset(self):
        self.nodes_mark.clear()
        self.next_hop.clear()
        
    def choose_nexthop(self, base_station, node, get_neighbors_func = get_neighbors, marker = 1):
        
        neighbors = get_neighbors_func(node, self.manager.all_nodes)
        self.nodes_mark[node] = marker
        marker += 1
        if D(node, base_station) <= radio_range:
            # move to phase 2
            self.next_hop[node] = base_station
            return True
        
        neighbors = sorted(neighbors, key=lambda x: D(x, base_station), reverse=True)
        
        # print(f"Choosing nexthop for {node.id}:")
        for neighbor in neighbors:
            # print(neighbor.id)
            if self.nodes_mark.get(neighbor) is not None:
                continue
            
            if self.choose_nexthop(base_station, neighbor, get_neighbors_func, marker= marker):
                self.next_hop[node] = neighbor
                return True
        
        # print(f"Mark {node.id} as block")
        
        return False
            



