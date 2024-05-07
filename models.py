from settings import PARAMS
    
class Node():
    """
    A wireless sensor node.
    """
    def __init__(self,  id, coordinates, energy = PARAMS.get('initial_energy')) -> None:
        # identification
        self.id = id
        # remaining energy of the node
        self.coordinates = coordinates
        # coordinates of the node, in [x,y] format
        self.energy = energy
        
        self.dead_flag = False
        
    @property
    def is_dead(self):
        """
        Returns the dead flag. The node is dead if it has no energy.
        """
        if self.energy <= 0:
            self.dead_flag = True
        return self.dead_flag
        
class Sink(Node):
    """
    Special node that acts as the base station for communicating with Users / Servers
    """
    def __init__(self, coordinates, energy = PARAMS.get('initial_energy') * 1000):
        # identification
        self.id = -1
        # remaining energy of the node
        self.energy = energy
        # coordinates of the node, in [x,y] format
        self.coordinates = coordinates
        
        self.dead_flag = False
        
    @property
    def is_dead(self):
        return False