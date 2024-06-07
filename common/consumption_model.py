from common.settings import PARAMS
import math
from enum import Enum
from utils.common import D
import random

class MSG_TYPE(Enum):
    HELLO = 100  
    DATA = 20000 * 120
    
E_elec = PARAMS.get('E_elec')
E_fs = PARAMS.get('E_fs')
E_da = PARAMS.get('E_da')
E_mp = PARAMS.get('E_mp')
p = PARAMS.get('p')
d0 = E_fs / E_mp
    
def send_receive_packets(sender, receiver, message_type):
    """
    Perform a transmission from one to another node in the simulation. 
    """
    send_energy_cost = E_Tx(message_type.value, D(sender, receiver))
    sender.energy -= send_energy_cost
            
    receive_energy_cost = E_Rx(message_type.value)
    receiver.energy -= receive_energy_cost

def send_receive_multiple_packets(sender, receivers, message_type):
    """
    Send packets of the same type from one to many nodes
    """
    for receiver in receivers:
        send_receive_packets(sender, receiver, message_type)

def cluster_head_data_fusion(node):
    node.energy -= PARAMS.get('E_da') * MSG_TYPE.DATA.value

def E_Tx(k,d):
    """
    Energy consumption for transmitting k bits over d meters
    """
    # if k < d0:
    return E_elec * k + E_fs * k * ( d ** 2 )
    # return E_elec * k + E_mp * k * d ** 4

def E_Rx(k):
    """
    Energy consumption for receiving k bits
    """
    return E_elec * k

round_duration = 2 # minutes
EHmax = 0.00008
charge_effi = PARAMS.get('charge_efficiency')
EH_E0 = PARAMS.get('eh_initial_energy')

class EnergyHarvesting():
    
    def __init__(self, time_offset = 0):
        self.current_time = time_offset
        self.eh_nodes = []    
        self.harvested_energy = 0
        self.old_energy_state = {}
    
    def pick_eh_nodes(self, ratio, all_nodes):
        random.seed(10) 
        while len(self.eh_nodes) < len(all_nodes) * ratio:
            temp_rand = random.randint(0, len(all_nodes) - 1)
            print(temp_rand)
            if all_nodes[temp_rand] in self.eh_nodes:
                continue
            self.eh_nodes.append(all_nodes[temp_rand])
            all_nodes[temp_rand].energy = EH_E0
            
    def tick(self):
        self.current_time += round_duration
        self.harvested_enery = self.get_harvested_energy()

    def save_energy_states(self):
        for node in self.eh_nodes:
            self.old_energy_state[node] = node.energy
        
    def use_or_store_harvested_energy(self):
        for node in self.eh_nodes:
            consumed_energy = self.old_energy_state.get(node) - node.energy
            if self.harvested_energy > consumed_energy: 
                node.energy += charge_effi * (self.harvested_energy - consumed_energy)
            else:
                node.energy += self.harvested_energy
            if node.energy > EH_E0:
                node.energy = EH_E0
        
    def is_eh_node(self, node):
        return node in self.eh_nodes
        
    def get_harvested_energy(self):
        hour =  int(self.current_time / 60) % 24
        if hour <= 7:
            return 0
        elif hour <= 10:
            return EHmax * (hour - 7) / 3
        elif hour <= 15:
            return EHmax * (1.2 - 0.2 * hour)
        elif hour <= 18:
            return EHmax * (18 - hour) / 3
        else:
            return 0