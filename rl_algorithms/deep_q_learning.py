from collections import namedtuple
from itertools import count
import random
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden = 50):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc3_bn = nn.BatchNorm1d(hidden)
        self.fc4 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        return self.fc4(x)


class DeepQLearningAgent:

    def __init__(self, environment, hidden = 50):
        self.environment = environment
        self.epsilon = 0.2
        self.batch_size = 64
        self.gamma = 1.0
        self.target_update = 5
        self.memory = ReplayMemory(1000000)
        self.timestep = 0

        self.state_space_dims = environment.state_space_dims
        self.n_actions = environment.n_actions
        self.actions = environment.action_space.values

        self.policy_net = DQN(self.state_space_dims, self.n_actions, hidden = hidden)
        self.target_net = DQN(self.state_space_dims, self.n_actions, hidden = hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.train() #train mode (train vs eval mode)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00001) #todo pokusaj nesto drugo
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def get_available_actions(self, disturbance):
        available_actions = []
        less_than_zero = list(filter(lambda x: x <= 0, self.actions))
        if self.timestep < 3:
            available_actions = list(filter(lambda x: x <= 0, self.actions))
        elif self.timestep < 6:
            available_actions = list(filter(lambda x: x >= 0, self.actions))
        else:
            available_actions = [0.0]
        if (len(available_actions) == 0):
            print('Warning in deep_q_learning.py -> get_action: No avaliable actions')
        return available_actions

    def get_action(self, state, epsilon, disturbance):
        #currently all action are available
        available_actions = self.get_available_actions(disturbance)
        if random.random() > epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                sorted_action_ids = self.policy_net(state).sort(-1, descending = True)[1].tolist()[0]
                action = None #todo remove later
                for sorted_action_id in sorted_action_ids:
                    if self.actions[sorted_action_id] in available_actions:
                        action = self.actions[sorted_action_id]
                        break
                if action is None:
                    print ('Error in deep_q_learning.py -> get_action: no action from sorted actions is selected')
                return action
                #action_index = self.policy_net(state).max(1)[1].view(1, 1)
                #self.policy_net.train()
                #return self.actions[action_index]
        else:
            action = random.choice(available_actions)
            return action   
            #return self.actions[random.randint(0, len(self.actions)-1)] 

    def train(self, n_episodes):
        total_episode_rewards = []
        collectPlotData = False
        #todo delete
        #self.policy_net.load_state_dict(torch.load("policy_net"))
        for i_episode in range(n_episodes):
            if (i_episode % 500 == 0):
                print("=========Episode: ", i_episode)
            #if (i_episode == int(0.02 * n_episodes)):
                #self.epsilon = 0.1
            done = False

            #state initialization... look at vvo project if necessary
            initial_disturbance = random.uniform(self.environment.min_disturbance, self.environment.max_disturbance)
            #print('initial_disturbance', initial_disturbance)
            state = self.environment.reset(initial_disturbance)

            state = torch.tensor([state], dtype=torch.float)
            total_episode_reward = 0
            self.timestep = 0

            while not done:
                #print("state", state)
                action = self.get_action(state, self.epsilon, self.environment.disturbance)
                #print("action", action)
                if (abs(action) > self.environment.high_set_point):
                    print('Warning: deep_q_learning.py: invalid action value: ', action)
                next_state, reward, done, _, _ = self.environment.step(action, collectPlotData)
                self.timestep += 1
                total_episode_reward += reward
                reward = torch.tensor([reward], dtype=torch.float)
                action = torch.tensor([action], dtype=torch.float)
                next_state = torch.tensor([next_state], dtype=torch.float)

                if done:
                    next_state = None
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()

            if (i_episode % 500 == 0):
                print ("total_episode_reward: ", total_episode_reward)

            total_episode_rewards.append(total_episode_reward)
            
            if (i_episode % 5000 == 0):
                time.sleep(60)
                torch.save(self.policy_net.state_dict(), "policy_net")

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        torch.save(self.policy_net.state_dict(), "policy_net")

        x_axis = [1 + j for j in range(len(total_episode_rewards))]
        plt.plot(x_axis, total_episode_rewards)
        plt.xlabel('Episode number') 
        plt.ylabel('Total episode reward') 
        plt.savefig("total_episode_rewards.png")
        plt.show()


    def test(self, test_sample_list):
        print('***********TEST***********')
        total_episode_reward_list = [] 
        self.policy_net.load_state_dict(torch.load("policy_net"))
        self.policy_net.eval()
        collectPlotData = True

        test_sample_id = 1
        for initial_disturbance in test_sample_list:
            print('Initial disturbance:', initial_disturbance)

            freqs = []
            rocofs = []
            control_efforts = [0 for i in range(25)]
            action_sums = [0 for i in range(25)]
            action_sum = 0
            state  = self.environment.reset(initial_disturbance)

            state = torch.tensor([state], dtype=torch.float)
            done = False
            total_episode_reward = 0

            self.timestep = 0
            #actions = [0.01, 0.02, 0.03, 0.0] #zadnja se nece gledati, samo da kod ne pukne
            while not done:
                #action = actions[i]
                action = self.get_action(state, epsilon = 0.0, disturbance = self.environment.disturbance)
                action_sum += action
                print('action',action)
                next_state, reward, done, temp_freqs, temp_rocofs = self.environment.step(action, collectPlotData)
                self.timestep += 1
                if not done:
                    #todo for more resources we should unpack the list of lists of freqs
                    freqs = temp_freqs #we override the freqs by the last results
                    rocofs = temp_rocofs
                    control_efforts += [action for i in range(25)]
                    action_sums += [action_sum for i in range(25)]
                    #print(control_efforts)
                    print('New disturbance:', self.environment.disturbance)
                    print('New freq:', self.environment.freq)
                    print('New rocof:', self.environment.rocof)
                    print('Reward:', reward)
                    print('**********************')

                total_episode_reward += reward
                state = torch.tensor([next_state], dtype=torch.float)
            plot_results(test_sample_id, freqs, rocofs, control_efforts, action_sums)
            test_sample_id += 1
            total_episode_reward_list.append(total_episode_reward)
        print ("Test set reward ", sum(total_episode_reward_list))

        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        #converts batch array of transitions to transiton of batch arrays
        batch = Transition(*zip(*transitions))

        #compute a mask of non final states and concatenate the batch elements
        #there will be zero q values for final states later... therefore we need mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype = torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1,1)
        reward_batch = torch.cat(batch.reward).view(-1,1)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select
        # the columns of actions taken. These are the actions which would've
        # been taken for each batch state according to policy net
        action_indices = (action_batch / self.environment.action_space.step).round() + self.environment.action_space.size // 2
        action_indices = action_indices.to(dtype=torch.int64)
        state_action_values = self.policy_net(state_batch).gather(1, action_indices)

        #gather radi isto sto i:
        #q_vals = []
        #for qv, ac in zip(Q(obs_batch), act_batch):
        #    q_vals.append(qv[ac])
        #q_vals = torch.cat(q_vals, dim=0)

        # Compute V(s_{t+1}) for all next states
        # q values of actions for non terminal states are computed using
        # the older target network, selecting the best reward with max
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() #manje od 128 stanja, nema final stanja
        #za stanja koja su final ce next_state_values biti 0
        #detach znaci da se nad varijablom next_state_values ne vrsi optimizacicja
        next_state_values = next_state_values.view(-1,1)
        # compute the expected Q values
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch

        #Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        #todo razmisli
        #for param in self.policy_net.parameters():
            #param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


def plot_results(test_sample_id, freqs, rocofs, control_efforts, action_sums):
    time = [i for i in range(len(freqs))]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.plot(time, freqs, label='Freq', color='g')
    ax1.set_title('Frequency')
    #ax1.legend(loc='upper right')
    #ax1.xlabel('s')  todo kako ovo uraditi za ax1?
    #ax1.ylabel('Hz') 

    ax2.plot(time, rocofs, label='Rocof', color='r')
    ax2.set_title('Rocof')
    #ax2.legend(loc='upper right')
    #ax2.xlabel('s') 

    ax3.plot(time, control_efforts, label='Control effort', color='k')
    ax3.set_title('Control effort')
    #ax3.legend(loc='upper right')
    #ax3.xlabel('s') 
    #ax3.ylabel('p.u.') 
    
    ax4.plot(time, action_sums, label='Action sums', color='b')
    ax4.set_title('Action sums')
    #ax4.legend(loc='upper right')

    fig.savefig(str(test_sample_id) + '_resuts.png')
    plt.show()