from simulator import SimManager
from consumption_model import send_receive_packets, MSG_TYPE, EnergyHarvesting
from settings import PARAMS
from sleep_scheduler import EC_CKN
from routing_protocol import TPGFRouter

# initiallize the network

manager = SimManager()
manager.make_sink()
manager.make_source()
manager.generate_nodes(seed=1556)

router = TPGFRouter(manager)
sleep_scheduler = EC_CKN(manager)
sleep_scheduler.k = 1

eh_model = EnergyHarvesting(time_offset=10*60)
eh_model.pick_eh_nodes(ratio = PARAMS.get('eh_ratio'), all_nodes = manager.all_nodes)

network_size = PARAMS.get('sensor_network_size')

D_Target  = PARAMS.get('D_Target')

# lifeime metrics
current_round = 0
fnd = 0

while not manager.all_node_dead:
    
    print(f"Round {current_round}")
    eh_model.tick()
    eh_model.save_energy_states()
    
    sleep_scheduler.k = 1
    sleep_scheduler.reset()
    sleep_scheduler.perform_scheduling()

    D = sleep_scheduler.get_coverage_degree(network_size = network_size, all_nodes = manager.all_nodes)
    while D < D_Target and sleep_scheduler.k < D_Target * 10:
        sleep_scheduler.k += 1
        sleep_scheduler.reset()
        sleep_scheduler.perform_scheduling()
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
        if node.is_dead and fnd == 0:
            fnd = current_round
    
    if fnd != 0:
        print(f"FND: {fnd}")
        break
    """
    for node in manager.all_nodes:
        if node.is_dead and fnd == 0:
            fnd = current_round
        if node.is_dead:
             continue
        send_receive_packets(node, manager.sink_node, MSG_TYPE.DATA)
    """
    
    current_round += 1
    
    
print(f"LND: {current_round}")