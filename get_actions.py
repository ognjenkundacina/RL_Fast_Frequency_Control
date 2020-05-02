import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable

import numpy as np
from collections import deque

'''
fajl state.txt
-0.09975962 f[0] tj VSC1...
-0.08151885 f[1]
-0.05560614 f[2]
-0.10383657 f[3]
-1.54993616 rf[0]
-1.30139484 rf[1]
-0.97959464 rf[2]
-1.65140661 rf[3]
0.0 agent step number (koji pocinje od 0) podjeljen sa 10. 
Moguce vrijednosti: 0.0, 0.1, 0.2, ..., 0.9

fajl actions.txt
0.05 VSC1
0.01 VSC2
'''

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1a = nn.Linear(hidden_size, hidden_size)
        #self.linear1a_bn = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        #self.linear2_bn = nn.BatchNorm1d(hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        #self.linear2_2bn = nn.BatchNorm1d(hidden_size)
        self.linear2_3 = nn.Linear(hidden_size, hidden_size)
        #self.linear2_3bn = nn.BatchNorm1d(hidden_size)
        self.linear2_4 = nn.Linear(hidden_size, hidden_size)
        #self.linear2_4bn = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        init_w = 3e-3
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear1a(x)) 
        x = torch.relu(self.linear2(x))       
        x = torch.relu(self.linear2_2(x))
        x = torch.relu(self.linear2_3(x))
        x = torch.relu(self.linear2_4(x))
        x = torch.tanh(self.linear3(x))
        return x


def reverse_action(action, action_space_high, action_space_low):
    act_k = (action_space_high - action_space_low)/ 2.
    act_b = (action_space_high + action_space_low)/ 2.
    return act_k * action + act_b  

state = []
#file_in = open('state.txt', 'r')
#for y in file_in.read().split('\n'):
    #state.append(float(y))
    
with open('state.txt') as f:
    for line in f:
        elems = line.strip()
        elems = elems.split(",")
        for elem in elems:
            state.append(float(elem))

print(state)
actor = Actor(9, 128, 2)
actor.load_state_dict(torch.load("model_actor"))

state = np.asarray(state)
state = Variable(torch.from_numpy(state).float().unsqueeze(0))
actor.eval()
action = actor.forward(state)
action = action.tolist()[0]
#output from actor network is normalized so:
action[0] = reverse_action(action[0], 0.1, 0.0)
action[1] = reverse_action(action[1], 0.1, 0.0)

file_out = open('actions.txt', 'w')
for act in action:
    file_out.write(str(act) + '\n')