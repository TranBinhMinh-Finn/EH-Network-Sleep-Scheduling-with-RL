from common.simulator import SimManager
from common.consumption_model import send_receive_packets, MSG_TYPE, EnergyHarvesting
from common.settings import PARAMS
# from sleep_scheduler import EC_CKN, EH_EC_CKN
from sleep_scheduler.RL_EH_EC_CKN import RL_EH_EC_CKN
from sleep_scheduler.EH_EC_CKN import EH_EC_CKN
from common.routing_protocol import TPGFRouter
import utils.plotter as plotter
# initiallize the network

import torch
from matplotlib import pyplot as plt
import math
import utils.logger as logger

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manager = SimManager()
manager.make_sink()
manager.make_source()
manager.generate_nodes(seed=1556)

eh_model = EnergyHarvesting(time_offset=10*60)
eh_model.pick_eh_nodes(ratio = PARAMS.get('eh_ratio'), all_nodes = manager.all_nodes)

router = TPGFRouter(manager)
sleep_scheduler = EH_EC_CKN(manager, eh_model.eh_nodes)
# sleep_scheduler = EC_CKN(manager)
sleep_scheduler.k = 1

network_size = PARAMS.get('sensor_network_size')

D_Target  = PARAMS.get('D_Target')

start_time = time.time()
logger.info('MAIN', 'START')

# lifeime metrics
current_round = 0
fnd = 0
"""
# eh-ec-ckn round

while not manager.all_node_dead:
    
    # print(f"Round {current_round}")
    eh_model.tick()
    eh_model.save_energy_states()
    
    sleep_scheduler.k = 1
    sleep_scheduler.reset()
    sleep_scheduler.perform_scheduling(sleep_scheduler.get_neighbors)

    D = sleep_scheduler.get_coverage_degree(network_size = network_size, all_nodes = manager.all_nodes)
    while D < D_Target and sleep_scheduler.k < D_Target * 10:
        sleep_scheduler.extend_eh_nodes_range = True
        D = sleep_scheduler.get_coverage_degree(network_size = network_size, all_nodes = manager.all_nodes)
        if D >= D_Target:
           break
        sleep_scheduler.k += 1
        sleep_scheduler.reset()
        sleep_scheduler.perform_scheduling(sleep_scheduler.get_neighbors)
        D = sleep_scheduler.get_coverage_degree(network_size = network_size, all_nodes = manager.all_nodes)
        
    
    if D < D_Target:
        print("Fail to satisfy required coverage.")
        break
    
    router.reset()
    router.choose_nexthop(base_station=manager.sink_node, node=manager.source_node, get_neighbors_func=sleep_scheduler.get_awake_neighbors)
    
    current_node = manager.source_node
    while current_node != manager.sink_node:
        next_hop = router.next_hop.get(current_node)
        if next_hop is None:
            print(f"No path from source node to sink. Stopping simulation")
            break
        # print(f"{current_node.id} -> {next_hop.id}")
        
        sender_energy = current_node.energy
        receiver_energy  = next_hop.energy
            
        send_receive_packets(current_node, next_hop, MSG_TYPE.DATA)
        
        current_node = next_hop
    
    eh_model.use_or_store_harvested_energy()
    
    if next_hop is not manager.sink_node:
        break
    
    for node in manager.all_nodes:
        if node.is_dead and fnd == 0 and node not in eh_model.eh_nodes:
            fnd = current_round
    
    if fnd != 0:
        print(f"FND: {fnd}")
        break
    
    current_round += 1
    
    
print(f"FND: {current_round}")
"""

import torch
from itertools import count

sleep_scheduler = RL_EH_EC_CKN(manager, eh_model.eh_nodes, device = device)

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

EH_E0 = PARAMS.get('eh_initial_energy')

episode_lifetimes = []
reward_sums = []
epsilons = []

for i in range(num_episodes):
    logger.info('TRAIN', f'Episode: {i}')
    fnd = 0
    manager.reset_nodes()
    for node in eh_model.eh_nodes:
        node.energy = EH_E0
    current_round = 0
    reward_sum = 0
    for current_round in count():
        logger.info('TRAIN', f'Round: {current_round}')
        coverage_reward = 1
        connectivity_reward = 0
        
        eh_model.tick()
        eh_model.save_energy_states()
        
        sleep_scheduler.perform_eh_nodes_sleep_scheduler(eh_model.harvested_energy)

        sleep_scheduler.k = 1
        sleep_scheduler.reset()
        sleep_scheduler.perform_scheduling(sleep_scheduler.get_neighbors)

        D = sleep_scheduler.get_coverage_degree(network_size = network_size, all_nodes = manager.all_nodes)
        while D < D_Target and sleep_scheduler.k < D_Target * 2:
            sleep_scheduler.k += 1
            sleep_scheduler.perform_scheduling(sleep_scheduler.get_neighbors)
            D = sleep_scheduler.get_coverage_degree(network_size = network_size, all_nodes = manager.all_nodes)
            if D >= D_Target:
                break
        
        if D < D_Target:
            # print("Fail to satisfy required coverage.")
            coverage_reward = 0
        
        logger.info('TRAIN', f'EC_CKN k value: {sleep_scheduler.k}')
        
        router.reset()
        router.choose_nexthop(base_station=manager.sink_node, node=manager.source_node, get_neighbors_func=sleep_scheduler.get_awake_neighbors)
        
        current_node = manager.source_node
        while current_node != manager.sink_node:
            next_hop = router.next_hop.get(current_node)
            if next_hop is None:
                print(f"No path from source node to sink. Stopping simulation")
                break
            # print(f"{current_node.id} -> {next_hop.id}")
            
            sender_energy = current_node.energy
            receiver_energy  = next_hop.energy
                
            send_receive_packets(current_node, next_hop, MSG_TYPE.DATA)
            
            current_node = next_hop
        
        eh_model.use_or_store_harvested_energy()
        
        if next_hop is not manager.sink_node:
            connectivity_reward = -1
            sleep_scheduler.update_model(eh_model.harvested_energy, reward = -2)
            # print(f"FND: {current_round}")
            model = sleep_scheduler.model.get(sleep_scheduler.eh_nodes[0])
            epsilon = model.eps_end + (model.eps_start - model.eps_end) * math.exp(-1. * model.steps_done / model.eps_decay)
            epsilons.append(epsilon)
            reward_sums.append(reward_sum)
            episode_lifetimes.append(current_round)
            break
        
        sleep_scheduler.update_model(eh_model.harvested_energy, reward = connectivity_reward + coverage_reward)
        
        reward_sum += connectivity_reward + coverage_reward
        for node in manager.all_nodes:
            if node.is_dead and fnd == 0 and node not in eh_model.eh_nodes:
                fnd = current_round
        
        if fnd != 0:
            print(f"FND: {fnd}")
            episode_lifetimes.append(current_round)
            model = sleep_scheduler.model.get(sleep_scheduler.eh_nodes[0])
            epsilon = model.eps_end + (model.eps_start - model.eps_end) * math.exp(-1. * model.steps_done / model.eps_decay)
            epsilons.append(epsilon)
            reward_sums.append(reward_sum)
            break
    
    logger.info('TRAIN', f'Lifetime: {current_round}')
    plotter.plot_durations(episode_lifetimes, reward_sums)
    
end_time = time.time()
logger.info('MAIN', f'COMPLETED IN: {end_time - start_time} seconds')
plotter.plot_durations(episode_lifetimes, reward_sums, show_result=True)
plt.ioff()
plt.show()




