from models import Node, Sink
from settings import PARAMS

import pandas as pd 
import numpy as np

import math
from utils import get_neighbors, D

radio_range = PARAMS.get('rr')

def generate_node_coordinates(nodes_number, x_size, y_size, seed = 42):
    """
    Generate Random coordinates
    """
    
    # X, Y = make_blobs(n_samples=nodes_number, 
    #                   centers=12, 
    #                   cluster_std=1.0, 
    #                   center_box=(-10, 10), 
    #                   shuffle=True, 
    #                   random_state=42, 
    #                   return_centers=False)
    
    np.random.seed(seed)
    X = np.random.randint(0, x_size, nodes_number)
    Y = np.random.randint(0, y_size, nodes_number)
    df = pd.DataFrame(data = zip(X,Y))
    # df = pd.DataFrame(data = X)
    # df /= df.abs().max()
    # df*=x_size/2
    # df+=x_size/2
    
    return df

class SimManager():
    all_nodes = []
    sink_node = None
    cluster_heads = []
    
    def generate_nodes(self, nodes_number = PARAMS.get('nodes_number'), generate_algo = generate_node_coordinates, seed = 42):
        """
        Generate Random Nodes in the simulation
        """
        coordinates = generate_algo(nodes_number, 
                          PARAMS.get('sensor_network_size')[0],
                          PARAMS.get('sensor_network_size')[1],
                          seed)
        
        for index, row in coordinates.iterrows():
            node = Node(id = len(self.all_nodes),
                        coordinates= [row[0], row[1]])
            self.all_nodes.append(node)
            
    def reset_nodes(self):
        for node in self.all_nodes:
            node.energy = PARAMS.get('initial_energy')
            node.dead_flag = False
            
    @property
    def all_node_dead(self):
        for node in self.all_nodes:
            if not node.is_dead:
                return False
        
        return True

    def dead_nodes_count(self):
        count = 0 
        for node in self.all_nodes:
            if node.is_dead:
                count += 1
        return count
    
    def make_sink(self, coordinates = PARAMS.get('base_station')):
        self.sink_node = Sink(coordinates=coordinates)
        
    def make_source(self, coordinates = PARAMS.get('source_node')):
        self.source_node = Node(id = -2, coordinates=coordinates, energy = PARAMS.get('initial_energy') * 1000)
        
    def get_neighbors(self, node, all_nodes, include_sink = True, include_source = True):
        neighbors = get_neighbors(node, all_nodes)
        if include_sink and D(node, self.sink_node) < radio_range:
            neighbors.append(self.sink_node)
        
        if include_source and D(node, self.source_node) < radio_range:
            neighbors.append(self.source_node)
            
        return neighbors
    
            