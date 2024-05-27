from common.settings import PARAMS
import math

radio_range = PARAMS.get('rr')

# if torch.cuda.is_available() else "cpu")

def D(node_q, node_s):
    """
    Distance between 2 nodes
    """
    q = node_q.coordinates
    s = node_s.coordinates
    return math.sqrt((q[0] - s[0]) ** 2 + (q[1] - s[1]) ** 2)

def get_neighbors(node, all_nodes):
    neighbors = []
    for node_ in all_nodes:
        if D(node_, node) <= radio_range and node.id != node_.id and not node_.is_dead:
            neighbors.append(node_)
            
    return neighbors

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

D_coverage = {}

def calculate_coverage_for_each_radius():
    """
    Calculate how many squares a quarter circle and its cutoff would cover based on the coverage requirement.
    """
    for r in range(radio_range, radio_range * 2 + 1):
        d_coverage = []
        d_coverage.append(0)
        for i in range(1, r + 1):
            j = r
            coverage = 0
            while euclidean_distance((i, j), (0,0)) > r:
                if euclidean_distance((i - 0.25, j - 0.25), (0,0)) <= r:
                    coverage += 1
                if euclidean_distance((i - 0.75, j - 0.25), (0,0)) <= r:
                    coverage += 1
                j-=1
            coverage += 4 * i * j
            d_coverage.append(coverage)
        D_coverage[r] = d_coverage
            

def calculate_coverage_radius(radius, top_cutoff, bottom_cutoff, left_cutoff, right_cutoff):
    """
    Calculate the coverage of a sensor node with their radius and cutoffs. A node has a cutoff if its sensing range reaches outside the network space.
    """
    coverage = 0
    D_matrix = D_coverage.get(radius)
    cutoffs = [[top_cutoff, bottom_cutoff], [left_cutoff, right_cutoff]]
    for cx in cutoffs[0]:
        for cy in cutoffs[1]: 
            if cx < radius and cy < radius:
                coverage += cx * cy * 4
            elif cx == radius:
                coverage += D_matrix[cy]
            elif cy == radius:
                coverage += D_matrix[cx]
    
    return coverage
        
def calculate_coverage(node, network_size, rr = radio_range):
    x, y = network_size
    x_node, y_node = node.coordinates    
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

calculate_coverage_for_each_radius()
        