import matplotlib.pyplot as plt
import numpy as np
from collections import deque

total_episode_rewards = []
with open('total_episode_rewards.txt') as f:
    for line in f:
        elems = line.strip()
        total_episode_rewards.append(float(elems))
        

x_axis = [1 + j for j in range(len(total_episode_rewards))]
plt.plot(x_axis, total_episode_rewards)
plt.xlabel('Episode number') 
plt.ylabel('Total episode reward') 
plt.show()
#plt.savefig("total_episode_rewards.png")
