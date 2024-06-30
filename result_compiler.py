from utils import plotter
import csv
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

def plot_values(title, data, xlabel, ylabel, window_size = 50):
    fig, ax = plt.subplots(1, layout = "constrained")
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim([np.min(data) - 10, np.max(data) + 10])
    ax.plot(data, color='green')
    
    moving_average = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    ax.plot(np.arange(window_size-1, len(data)), moving_average, label='Lifetime MA', color='red')
    
    plt.show()

def plot_coverage_over_lifetime(coverage, lifetime,  window_size = 50, ax = None, title = ""):
    if ax is None:
        fig, ax = plt.subplots(1, layout = "constrained")
        fig.set_figheight(5)
        fig.set_figwidth(10)
    
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Time Step')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    y_min = min(np.min(lifetime), np.min(coverage))
    y_min = max(y_min - 10, 0)
    # y_min = 0
    
    y_max = np.max(lifetime) + 10
    
    ax.set_ylim([y_min, y_max])
    
    ax.grid()
        
    ax.plot(coverage, color='orange', label = 'Số time step đạt coverage')
    ax.plot(lifetime, color='green', label = 'Thời gian sống')
    
    moving_average = np.convolve(coverage, np.ones(window_size)/window_size, mode='valid')
    ax.plot(np.arange(window_size-1, len(coverage)), moving_average, label='Trung bình động time step đạt coverage', color='red')
    moving_average = np.convolve(lifetime, np.ones(window_size)/window_size, mode='valid')
    ax.plot(np.arange(window_size-1, len(lifetime)), moving_average, label='Trung bình động thời gian sống', color='purple')
    
    # ax.legend(loc = 'lower right')
    # plt.show()

def plot_comparing_reward(dqn, gru, window_size = 50):
    fig, ax = plt.subplots(1, layout = "constrained")
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    # ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Tổng reward episode')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    y_min = min(np.min(dqn), np.min(gru)) - 10
    # y_min = 0
    
    y_max = max(np.max(dqn), np.max(gru)) + 10
    
    ax.set_ylim([y_min, y_max])
    
    ax.grid()
    
    ax.plot(dqn, label = 'Linear')
    ax.plot(gru, label = 'GRU')
    
    moving_average = np.convolve(dqn, np.ones(window_size)/window_size, mode='valid')
    ax.plot(np.arange(window_size-1, len(dqn)), moving_average, label='Trung bình động reward Linear', color='red')
    moving_average = np.convolve(gru, np.ones(window_size)/window_size, mode='valid')
    ax.plot(np.arange(window_size-1, len(gru)), moving_average, label='Trung bình động reward GRU', color='purple')
    
    ax.legend(loc = 'lower right')
    plt.show()
    
def parse_array(array_string):
    array_string = array_string.replace('[','')
    array_string = array_string.replace(']','')
    array = array_string.split()
    array = [float(x) for x in array]
    return array

headers = ['episode', 'lifetime', 'n_rounds_with_coverage', 'reward_sum', 'avg_loss']

def read_data(directory = os.path.join("res/dqn/episode")):
    data = {}

    for root,dirs,files in os.walk(directory):
        for file in files:
            parsed = file.split('_')
            map = int(parsed[2])
            rep = int(parsed[4])


            if parsed[-1] == 'reward.csv' and directory == os.path.join("res/dqn/episode"):
                continue
            
            with open(os.path.join(directory, file), 'r') as f:
                reader = csv.DictReader(f, 
                                        delimiter=',', 
                                        lineterminator='\n', 
                                        fieldnames=headers,
                                        )
                next(reader, None)
                for row in reader:
                    episode = row['episode']
                    for field in headers:
                        if field == 'episode':
                            continue
                        if data.get((field,map,rep)) is None:
                            data[(field,map,rep)] = []
                        if field != 'avg_loss':
                            data[(field,map,rep)].append(int((row[field])))
                        else:
                            avg = np.average(parse_array(row[field]))
                            data[(field,map,rep)].append(avg)
    return data

data1 = read_data(os.path.join("res/dqn/episode"))

data2 = read_data(os.path.join("res/gru/episode"))

dqn_reward = data1[('reward_sum',2,9)]

gru_reward = data2[('reward_sum',2,10)]

plot_comparing_reward(dqn_reward, gru_reward)

displaying_maps = [1,2,3,5,6,7]

fig, axs_ = plt.subplots(3, 2, layout = "constrained")

axs = []
for axs_row in axs_:
    for ax in axs_row:
        axs.append(ax)     

for map, it in zip(displaying_maps, range(1,7)):
    # coverage = np.zeros(len(data2[('n_rounds_with_coverage',map,1)]))
    # for i in range(1, 11):
    #     coverage = np.add(coverage, data2[('n_rounds_with_coverage',map,i)])
    # coverage = np.divide(coverage, 10)
    
    coverage = data2[('n_rounds_with_coverage',map,10)]
    
    lifetime = data2[('lifetime',map,10)]
    # lifetime = np.zeros(len(data2[('lifetime',map,1)]))
    # for i in range(1, 11):
    #     lifetime = np.add(lifetime, data2[('lifetime',map,i)])
    # lifetime = np.divide(lifetime, 10)
    
    plot_coverage_over_lifetime(coverage=coverage, lifetime=lifetime, ax=axs[it-1], title=f"Map {it}")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'size': 16})
plt.show()
    
    
# for map in range(1,8):
#     for field in headers:
#         if field == 'episode': # or field == 'avg_loss':
#             continue
#         sum = np.zeros(len(data1[(field,map,1)]))
#         for i in range(1, 11):
#             #if data1[(field,map,i)][-1] < sum[-1] / i:
#             #    continue
#             sum = np.add(sum, data1[(field,map,10)])
#         sum = np.divide(sum, 10)
#         plot_values(title=f"Map {map} {field}",
#                     data=sum,
#                     xlabel="Episode",
#                     ylabel=field,
#                     )
    
baseline = [26, 82, 92, 94, 97, 102]
result = [80, 99, 99, 99, 119, 116]
x_nodes = [1, 2, 4, 8, 10, 16]

fig, ax = plt.subplots(1, layout = "constrained")
fig.set_figheight(5)
fig.set_figwidth(8)


ax.set_xlabel('Số nút mạng có năng lượng tái tạo')
ax.set_ylabel('Lifetime')

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid()

ax.plot(x_nodes, baseline, label = 'baseline')
ax.plot(x_nodes, result, label = 'đề xuất')
ax.legend(loc="lower right")

plt.show()

                    
                
       
           