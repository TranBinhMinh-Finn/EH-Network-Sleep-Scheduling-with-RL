import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


fig, axs = plt.subplots(1, layout = "constrained")
fig.set_figheight(5)
fig.set_figwidth(10)
ax = axs
# ax2 = ax.twinx()


def plot_durations(episode_lifetimes, 
                   reward_sums, 
                   show_result=False):
    
    # plt.figure(1)
    
    global fig, ax, ax2
    if ax is None:
        ax = axs[0]
        # ax2 = ax.twinx()
        
    window_size = 50
    
    ax.clear()
    
    if show_result:
        ax.set_title('Result')
    else:
        ax.set_title('Training...')
        
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rounds')
    ax.plot(episode_lifetimes, color = 'orange', label = "Lifetime")
    
    if len(episode_lifetimes) >= window_size:
        moving_average = np.convolve(episode_lifetimes, np.ones(window_size)/window_size, mode='valid')
        ax.plot(np.arange(window_size-1, len(episode_lifetimes)), moving_average, label='Lifetime MA', color = 'purple')
        
    ax.plot(reward_sums, color = 'green', label = "Reward")
    
    if len(reward_sums) >= window_size:
        moving_average = np.convolve(reward_sums, np.ones(window_size)/window_size, mode='valid')
        ax.plot(np.arange(window_size-1, len(reward_sums)), moving_average, label='Reward MA', color = 'red')

    ax.legend(loc="lower left")
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            # plt.savefig('result.png')
            display.display(plt.gcf())
            

ax_epsilon = None

def plot_epsilon(epsilons, show_result = False):
    global ax_epsilon, ax
    if ax_epsilon is None:
        ax_epsilon =  axs[1]
    
    ax_epsilon.clear()
    # ax_epsilon.yaxis.set_label_position("right")
    ax_epsilon.set_xlabel('Episode')
    ax_epsilon.set_ylabel('Epsilon')
    ax_epsilon.plot(epsilons, label = 'Epsilon')
    
    ax_epsilon.legend(loc="upper left")
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            # plt.savefig('result.png')
            display.display(plt.gcf())
            
            
def plot_round(node_energy, show_result = False):
    global ax_state
    if ax_state is None:
        ax_state = axs[1]
    
    ax_state.clear()
    ax_state.set_xlabel('Round')
    ax_state.set_ylabel('Energy')
    for node, energy in node_energy.items():
        ax_state.plot(energy, label = f"Node {node.id}")
    
    # ax_state.legend(loc="lower left")
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
    if show_result:
        file_name = f"res/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"
        fig.savefig(file_name)

ax_loss = None

def plot_loss(node_energy, show_result = False):
    global ax_loss
    if ax_loss is None:
        ax_loss = axs[2]
    
    ax_loss.clear()
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss')
    for node, energy in node_energy.items():
        ax_loss.plot(energy, label = f"Node {node.id}")
    
    ax_loss.legend(loc="lower left")
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
    if show_result:
        file_name = f"res/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.png"
        # fig.savefig(file_name)