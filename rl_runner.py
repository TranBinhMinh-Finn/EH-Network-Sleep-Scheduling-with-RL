import torch
import time
from utils import logger
from matplotlib import pyplot as plt
from rl_modules.trainer import Trainer
import os
from datetime import datetime
from common import settings

from utils import plotter

start_time = time.time()
logger.info('MAIN', 'START')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

maps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 15/20]

if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

save_directory = 'res/gru'
    
for map in range(2,3):
    settings.PARAMS["eh_ratio"] = maps[map]
    
    for rep in range(10):
        logger.info('MAIN', f'Map: {map + 1}, Rep: {rep + 1}')
        filename = f"Map_{map + 1}_Rep_{rep + 1}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_negative_reward"
        
        episode_metrics_save_file = "Episode_" + filename + ".csv"
        timestep_metrics_save_file = "Timestep_" + filename + ".csv"
        model_save_file = "Model_" + filename + ".pth"

        trainer = Trainer(n_episode=num_episodes, 
                        device=device,
                        coordinate_seed = 10,
                        save_energy_states = False,
                        episode_metrics_save_file = os.path.join(save_directory, 'episode', episode_metrics_save_file),
                        timestep_metrics_save_file = os.path.join(save_directory, 'timestep', timestep_metrics_save_file),
                        model_save_file= os.path.join(save_directory, 'model', model_save_file)) 

        # all_nodes = trainer.env.manager.all_nodes
        # eh_nodes = trainer.env.eh_model.eh_nodes
        # 
        # coordinates = [node.coordinates for node in all_nodes]
        # 
        # eh = [int(node in eh_nodes) for node in all_nodes]
        
        # plotter.plot_coordinates(coordinates, eh)
        trainer.train()
        trainer.save_model()
        
        
end_time = time.time()

logger.info('MAIN', f'COMPLETED IN: {end_time - start_time} seconds')

# plt.ioff()
# plt.show()




