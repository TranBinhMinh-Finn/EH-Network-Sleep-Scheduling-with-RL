import torch
import time
from utils import logger
from matplotlib import pyplot as plt
from rl_modules.trainer import Trainer

start_time = time.time()
logger.info('MAIN', 'START')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50
    
trainer = Trainer(n_episode=num_episodes, 
                  device=device,
                  coordinate_seed = 10,
                  save_energy_states = False) 

trainer.train()

end_time = time.time()

logger.info('MAIN', f'COMPLETED IN: {end_time - start_time} seconds')

plt.ioff()
plt.show()




