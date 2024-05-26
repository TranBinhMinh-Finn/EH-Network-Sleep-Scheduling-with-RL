import matplotlib
from matplotlib import pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_durations(episode_lifetimes, reward_sums, epsilons, show_result=False):
    plt.figure(1)
    lifetimes_t = torch.tensor(episode_lifetimes, dtype=torch.int)
    if show_result:
        plt.clf()
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('FND')
    plt.plot(lifetimes_t.numpy())
    
    reward_sums_t = torch.tensor(reward_sums, dtype=torch.int)
    ax2 = plt.twinx()
    ax2.plot(reward_sums_t.numpy(), color='b', linestyle='--')
    ax2.set_ylabel('Reward')
    
    # epsilons_t = torch.tensor(epsilons, dtype=torch.float)
    # ax2 = plt.twinx()
    # ax2.plot(epsilons_t.numpy(), color='b', linestyle='--')
    # ax2.set_ylabel('Epsilon')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            plt.savefig('result.png')
            display.display(plt.gcf())
            