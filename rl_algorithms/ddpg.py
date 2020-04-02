import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable

import numpy as np
from collections import deque
import random
import time
import matplotlib.pyplot as plt

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1a = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_bn = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        init_w = 3e-3
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear1a(x))
        x = torch.relu(self.linear2_bn(self.linear2(x)))
        x = self.linear3(x) #returns q value, should not be limited by tanh
        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1a = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_bn = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        init_w = 3e-3
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear1a(x))
        x = torch.relu(self.linear2_bn(self.linear2(x)))        
        x = torch.tanh(self.linear3(x))
        return x

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.1, max_sigma=0.1, min_sigma=0.1, decay_period=100):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high

        self.reset()        

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t = 0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma-self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

#scales action from [-1, 1] to [-0.01, 0.4]
def reverse_action(action, action_space):
    act_k = (action_space.high - action_space.low)/ 2.
    act_b = (action_space.high + action_space.low)/ 2.
    return act_k * action + act_b  

#scales action from [-0.01, 0.4] to [-1, 1]
def normalize_action(action, action_space):
    act_k_inv = 2./(action_space.high - action_space.low)
    act_b = (action_space.high + action_space.low)/ 2.
    return act_k_inv * (action - act_b)

#scales action tensor from [-1, 1] to [-0.01, 0.4]
def reverse_action_tensor(action, action_space):
    high = np.asscalar(action_space.high[0])
    low = np.asscalar(action_space.low[0])
    act_k = (high - low)/ 2.
    act_b = (high + low)/ 2.
    act_b_tensor = act_b * torch.ones(action.shape)
    return act_k * action + act_b_tensor

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, environment, hidden_size=256, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.99, tau=1e-3, max_memory_size=600000):
        self.environment = environment
        self.num_states = environment.state_space_dims
        self.num_actions = environment.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        self.timestep = 0

        self.batch_size = 128

        self.hidden_size = hidden_size
        self.actor = Actor(self.num_states, self.hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, self.hidden_size, self.num_actions)
        self.actor.train()
        self.critic = Critic(self.num_states+self.num_actions, self.hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states+self.num_actions, self.hidden_size, self.num_actions)
        self.critic.train()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        self.replay_buffer = ReplayBuffer(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        self.noise = OUNoise(self.environment.action_space)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        self.actor.eval()
        action = self.actor.forward(state)
        self.actor.train()
        action = action.tolist()[0]
        #output from actor network is normalized so:
        action = reverse_action(action, self.environment.action_space)
        return action

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        #print('states', states)
        #print('actions', actions)

        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        #output from actor network is normalized so:
        next_actions = reverse_action_tensor(next_actions, self.environment.action_space)
        
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + (1.0 - dones) * self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime.detach())

        #actor loss
        next_actions_pol_loss = self.actor.forward(states)
        next_actions_pol_loss = reverse_action_tensor(next_actions_pol_loss, self.environment.action_space)
        policy_loss = -self.critic.forward(states, next_actions_pol_loss).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

    
    def train(self, n_episodes):
        #self.actor.load_state_dict(torch.load("model_actor"))
        #self.critic.load_state_dict(torch.load("model_critic"))
        total_episode_rewards = []
        collectPlotData = False
        for i_episode in range(n_episodes):
            if (i_episode % 100 == 0):
                print("Episode: ", i_episode)
                
            if (i_episode == 70000):
                self.noise.min_sigma = 0.02
                self.noise.max_sigma = 0.02

            initial_disturbance = random.uniform(self.environment.min_disturbance, self.environment.max_disturbance)
            #print('initial_disturbance', initial_disturbance)

            node_ids = range(1, 40) #1, 2,... 39
            values = [0.0 for i in range(len(node_ids))]
            initial_disturbance_dict = dict(zip(node_ids, values))
            initial_disturbance_dict[16] = initial_disturbance
            state = self.environment.reset(initial_disturbance_dict)

            self.noise.reset()
            done = False
            episode_iterator = 0
            total_episode_reward = 0 
            self.timestep = 0

            while not done:
                state = np.asarray(state)
                action = self.get_action(state)
                #print ("train action ", action)
                action = self.noise.get_action(action, self.timestep)
                #print ("train action with noise", action)
                next_state, reward, done, _, _ = self.environment.step(action.tolist(), collectPlotData)
                self.timestep += 1
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.update()

                state = next_state
                total_episode_reward += reward

            total_episode_rewards.append(total_episode_reward)
            
            if (i_episode % 100 == 0):
                print ("total_episode_reward: ", total_episode_reward)
            
            if (i_episode % 1000 == 0):
                #time.sleep(60)
                torch.save(self.actor.state_dict(), "./trained_nets/model_actor" + str(i_episode))
                torch.save(self.critic.state_dict(), "./trained_nets/model_critic" + str(i_episode))                
        
        torch.save(self.actor.state_dict(), "model_actor")
        torch.save(self.critic.state_dict(), "model_critic")
        
        x_axis = [1 + j for j in range(len(total_episode_rewards))]
        plt.plot(x_axis, total_episode_rewards)
        plt.xlabel('Episode number') 
        plt.ylabel('Total episode reward') 
        plt.savefig("total_episode_rewards.png")
        #plt.show()
        
        
    def train_with_weight_averaging(self, n_episodes):
        self.actor.load_state_dict(torch.load("./pretrained_weights/model_actor"))
        self.critic.load_state_dict(torch.load("./pretrained_weights/model_critic"))
        
        self.weight_averaging_period = 1
        self.n_swa = 1
        self.actor_swa = Actor(self.num_states, self.hidden_size, self.num_actions)
        self.critic_swa = Critic(self.num_states+self.num_actions, self.hidden_size, self.num_actions)
        
        for swa_param, param in zip(self.actor_swa.parameters(), self.actor.parameters()):
            swa_param.data.copy_(param.data)

        for swa_param, param in zip(self.critic_swa.parameters(), self.critic.parameters()):
            swa_param.data.copy_(param.data)
        
        total_episode_rewards = []
        collectPlotData = False
        for i_episode in range(n_episodes):
            if (i_episode % 100 == 0):
                print("Episode: ", i_episode)

            initial_disturbance = random.uniform(self.environment.min_disturbance, self.environment.max_disturbance)
            #print('initial_disturbance', initial_disturbance)

            node_ids = range(1, 40) #1, 2,... 39
            values = [0.0 for i in range(len(node_ids))]
            initial_disturbance_dict = dict(zip(node_ids, values))
            initial_disturbance_dict[16] = initial_disturbance
            state = self.environment.reset(initial_disturbance_dict)

            self.noise.reset()
            done = False
            episode_iterator = 0
            total_episode_reward = 0 
            self.timestep = 0

            while not done:
                state = np.asarray(state)
                action = self.get_action(state)
                #print ("train action ", action)
                action = self.noise.get_action(action, self.timestep)
                #print ("train action with noise", action)
                next_state, reward, done, _, _ = self.environment.step(action.tolist(), collectPlotData)
                self.timestep += 1
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.update()

                state = next_state
                total_episode_reward += reward

            total_episode_rewards.append(total_episode_reward)
            
            if (i_episode % 100 == 0):
                print ("total_episode_reward: ", total_episode_reward)
                
            if (i_episode % self.weight_averaging_period == 0):
                for swa_param, param in zip(self.actor_swa.parameters(), self.actor.parameters()):
                    swa_param.data.copy_( (swa_param.data*self.n_swa + param.data*1.0) / (1.0 * (self.n_swa + 1)))
                for swa_param, param in zip(self.critic_swa.parameters(), self.critic.parameters()):
                    swa_param.data.copy_( (swa_param.data*self.n_swa + param.data*1.0) / (1.0 * (self.n_swa + 1)))
                self.n_swa += 1
                #torch.save(self.actor_swa.state_dict(), "./trained_nets/actor_swa" + str(i_episode))
                #torch.save(self.critic_swa.state_dict(), "./trained_nets/critic_swa" + str(i_episode))
            
            if (i_episode % 1000 == 0):
                time.sleep(60)
                torch.save(self.actor.state_dict(), "./trained_nets/model_actor" + str(i_episode))
                torch.save(self.critic.state_dict(), "./trained_nets/model_critic" + str(i_episode))                
        
        torch.save(self.actor_swa.state_dict(), "model_actor")
        torch.save(self.critic_swa.state_dict(), "model_critic")
        
        x_axis = [1 + j for j in range(len(total_episode_rewards))]
        plt.plot(x_axis, total_episode_rewards)
        plt.xlabel('Episode number') 
        plt.ylabel('Total episode reward') 
        plt.savefig("total_episode_rewards.png")
        #plt.show()

    def test(self, test_sample_list):
        print('***********TEST***********')
        total_episode_reward_list = [] 
        collectPlotData = True
        
        test_sample_id = 1
        self.actor.load_state_dict(torch.load("model_actor"))

        for initial_disturbance_dict in test_sample_list:
            freqs = []
            rocofs = []
            control_efforts = [ [0 for i in range(25)], [0 for i in range(25)], [0 for i in range(25)] ]
            action_sums = [ [0 for i in range(25)], [0 for i in range(25)], [0 for i in range(25)] ]
            action_sum = [0 for i in range(self.num_actions)] # = [0, 0, 0]
            state  = self.environment.reset(initial_disturbance_dict)
            done = False
            total_episode_reward = 0

            self.timestep = 0
            while not done:
                state = np.asarray(state)
                action = self.get_action(state)
                #self.actor.eval()

                action_sum += action
                print('action',action)
                next_state, reward, done, temp_freqs, temp_rocofs = self.environment.step(action, collectPlotData)
                self.timestep += 1
                #if not done:
                #todo for more resources we should unpack the list of lists of freqs
                all_freqs = temp_freqs #we override the freqs by the last results
                all_rocofs = temp_rocofs
                control_efforts[0] += [action[0] for i in range(25)]
                control_efforts[1] += [action[1] for i in range(25)]
                control_efforts[2] += [action[2] for i in range(25)]
                action_sums[0] += [action_sum[0] for i in range(25)]
                action_sums[1] += [action_sum[1] for i in range(25)]
                action_sums[2] += [action_sum[2] for i in range(25)]
                print('Reward:', reward)
                print('**********************')

                total_episode_reward += reward
                state = next_state
            plot_results(test_sample_id, all_freqs, all_rocofs, control_efforts, action_sums)
            test_sample_id += 1
            total_episode_reward_list.append(total_episode_reward)
        print ("Test set reward ", sum(total_episode_reward_list))


def plot_results(test_sample_id, all_freqs, all_rocofs, all_control_efforts, all_action_sums):
    time = [i for i in range(len(all_control_efforts[0]))]

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, sharex=True)

    vsc_frequencies = all_freqs[:3]
    gen_frequencies = all_freqs[3:]

    ax0.set_title('VSC frequencies')
    i = 1
    for one_source_freqs in vsc_frequencies:
        one_source_freqs = [f + 50.0 for f in one_source_freqs]
        ax0.plot(time, one_source_freqs, label='Freq'+str(i))
        #ax0.plot(time, one_source_freqs, label='Freq', color='g')
        ax0.legend(loc='upper right')
        i += 1


    ax1.set_title('Generator frequencies')
    for one_source_freqs in gen_frequencies:
        one_source_freqs = [f + 50.0 for f in one_source_freqs]
        ax1.plot(time, one_source_freqs, label='Freq'+str(i))
        #ax1.plot(time, one_source_freqs, label='Freq', color='g')
        i += 1
        
        #ax1.legend(loc='upper right')
        #ax1.xlabel('s')  todo kako ovo uraditi za ax1?
        #ax1.ylabel('Hz') 

    ax2.set_title('Rocof')
    i = 1
    for one_source_rocofs in all_rocofs:
        ax2.plot(time, one_source_rocofs, label='Rocof'+str(i))
        i += 1
        #ax2.legend(loc='upper right')
        #ax2.xlabel('s') 

    ax3.set_title('Control effort')
    i = 1
    for one_vsc_control_efforts in all_control_efforts:
        ax3.plot(time, one_vsc_control_efforts, label='Control effort'+str(i))
        i += 1
        #ax3.legend(loc='upper right')
        #ax3.xlabel('s') 
        #ax3.ylabel('p.u.') 
    
    ax4.set_title('Action sums')
    i = 1
    for one_vsc_action_sums in all_action_sums:
        ax4.plot(time, one_vsc_action_sums, label='Action sums'+str(i))
        i += 1
        #ax4.legend(loc='upper right')

    fig.savefig(str(test_sample_id) + '_resuts.png')
    plt.show()