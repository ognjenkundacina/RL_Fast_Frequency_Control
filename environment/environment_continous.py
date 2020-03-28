import gym
from gym import spaces
import random
import numpy as np
from scipy import signal
import pickle
from gym.spaces import Tuple
from gym.spaces.space import Space
from config import *
import math
import os
from csv import reader

LOW_FREQ_LIMIT = -0.49
HIGH_FREQ_LIMIT = 0.49

#referent directions of active powers:
#negative disturbance - demand increase (frequency decrease)
#positive action - 'generation' increase

def load_matrix_from_csv(file_name):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './' + file_name)
    with open(file_path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        new_list_of_rows = []
        for row in list_of_rows:
            new_row = []
            for item in row:
                new_row.append(float(item))
            new_list_of_rows.append(new_row)
        return new_list_of_rows
        
class ScipyModel():
    def __init__(self):
        # Define the frequnecy response state-space
        A = load_matrix_from_csv("A_matrix.csv") #85 x 85
        B = load_matrix_from_csv("B_matrix.csv") #85 x 49
        C = load_matrix_from_csv("C_matrix.csv") #10 x 85
        D = load_matrix_from_csv("D_matrix.csv") #10 x 49
        sys = signal.StateSpace(A,B,C,D)

        # Define simulation parameters

        self.nSteps = 100 # Total number of time steps
        self.nDist = 5    # Disturbance time instant
        self.Td = 0.01     # Time discretization

        #N_ACTIONS_IN_SEQUENCE + 1 is the number of intervals in the episode (start contains no actions)
        self.n_agent_timestep_steps = self.nSteps // (N_ACTIONS_IN_SEQUENCE + 1)

        if self.nSteps % (N_ACTIONS_IN_SEQUENCE + 1) != 0:
            print('WARNING in environment_discrete.py: self.nSteps not divisible by (N_ACTIONS_IN_SEQUENCE + 1)')

        # Discrete state space
        sysd = sys.to_discrete(self.Td)
        self.Ad = np.array(sysd.A)
        self.Bd = np.array(sysd.B)
        self.Cd = np.array(sysd.C)
        self.Dd = np.array(sysd.D)
        self.x = np.zeros((85,self.nSteps))
        self.f = np.zeros((10,self.nSteps))
        self.rf = np.zeros((10,self.nSteps))
        self.u = np.zeros((49,self.nSteps)) #disturbance
        self.t = np.zeros((1,self.nSteps))

    def initialize_control_vector(self, initial_disturbance_dict):
        for i in range(49):
            if i<=19:
                self.u[:,self.nDist:self.nSteps] = 0
            else:
                node_id = i - 9 #Comment1, buttom of the file
                self.u[i ,self.nDist:self.nSteps+1] = initial_disturbance_dict[node_id]


    def reset_model(self, initial_disturbance_dict):
        self.x = np.zeros((85,self.nSteps+2))
        self.f = np.zeros((10,self.nSteps))
        self.rf = np.zeros((10,self.nSteps))
        self.u = np.zeros((49,self.nSteps+1))
        self.initialize_control_vector(initial_disturbance_dict)
        self.t = np.zeros((1,self.nSteps))

        for i in range (self.n_agent_timestep_steps):
            self.x[:,i+1] = np.dot(self.Ad, self.x[:,i]) + np.dot(self.Bd, self.u[:,i])
            self.f[:,i] = np.dot(self.Cd, self.x[:,i]) + np.dot(self.Dd, self.u[:,i])
            if i != 0:
                self.rf[:,i] = (self.f[:,i] - self.f[:,i-1])/self.Td
            self.t[:,i] = i*self.Td 
        
        #test
        if (i*self.Td != 0.24):
            print('WARNING in environment_discrete.py: i*self.Td != 0.24')

        #size of self.f[:,i].tolist() is 10
        #i is last step index in agents step (last step of the for loop)
        state = []
        state += self.f[:,i].tolist()
        state += self.rf[:,i].tolist()

        return state
            
    #agent timestep count starts from zero, but here, during the reset_model, 
    #the amount of n_agent_timestep_steps has been done
    def get_next_state(self, agent_timestep, action, collectPlotData):
        current_step = self.n_agent_timestep_steps * (agent_timestep + 1)
        num_of_violated_freqs = 0
        is_freq_violated = [0 for i in range(10)]
        for i in range(current_step, current_step + self.n_agent_timestep_steps):
            self.x[:,i+1] = np.dot(self.Ad, self.x[:,i]) + np.dot(self.Bd, self.u[:,i])
            self.f[:,i] = np.dot(self.Cd, self.x[:,i]) + np.dot(self.Dd, self.u[:,i])
            if i != 0:
                self.rf[:,i] = (self.f[:,i] - self.f[:,i-1])/self.Td
            self.t[:,i] = i*self.Td 
            
            self.u[0,i+1] = self.u[0,current_step - 1] + action[0] #Comment1, buttom of the file
            self.u[1,i+1] = self.u[1,current_step - 1] + action[1]
            self.u[2,i+1] = self.u[2,current_step - 1] + action[2]

            for j in range(10):
                if (self.f[j,i] > HIGH_FREQ_LIMIT) or (self.f[j,i] < LOW_FREQ_LIMIT):
                    is_freq_violated[j] = 1

        num_of_violated_freqs = sum(is_freq_violated)

        #size of self.f[:,i].tolist() is 10
        #i is last step index in agents step (last step of the for loop)
        next_state_freq = self.f[:,i].tolist()
        next_state_rocof = self.rf[:,i].tolist()

        if (collectPlotData and i == self.nSteps - 1):
            #10 lists of frequences
            freqs_all_steps = self.f.tolist()
            rocofs_all_steps = self.rf.tolist()
            #print(self.u)
        else:
            freqs_all_steps = []
            rocofs_all_steps = []

        return next_state_freq, next_state_rocof, freqs_all_steps, rocofs_all_steps, num_of_violated_freqs


class EnvironmentContinous(gym.Env):
    
    def __init__(self):
        super(EnvironmentContinous, self).__init__()
        
        self.freq = 0
        self.rocof = 0
        self.timestep = 0
        self.state = (self.freq, self.rocof, self.timestep / float(N_ACTIONS_IN_SEQUENCE) )
        ####self.state = (self.freq, self.rocof)
        ##########self.disturbance = 0

        self.min_disturbance = 0.0
        self.max_disturbance = 2.0

        self.state_space_dims = 21 #f i rocof i timestep
        ####self.state_space_dims = 2 #f i rocof i timestep
        self.action_space_dims = 3 #delta P
        self.action_sum = [0 for i in range(self.action_space_dims)] #models setpoint change, that should be zero at the end

        self.low_set_point = -0.01
        self.high_set_point = 0.3
        low_action_limit = [self.low_set_point for i in range(self.action_space_dims)]
        high_action_limit = [self.high_set_point for i in range(self.action_space_dims)]
        self.action_space = spaces.Box(low=np.array(low_action_limit), high=np.array(high_action_limit), dtype=np.float16)

        self.scipy_model = ScipyModel()

        self.low_freq_limit = LOW_FREQ_LIMIT
        self.high_freq_limit = HIGH_FREQ_LIMIT

    def update_state(self, action, collectPlotData):

        next_state_freq, next_state_rocof, freqs_all_steps, rocofs_all_steps, num_of_violated_freqs = self.scipy_model.get_next_state(self.timestep, action, collectPlotData)

        self.state = []
        self.state += next_state_freq
        self.state += next_state_rocof
        self.timestep += 1
        self.state.append(self.timestep / float(N_ACTIONS_IN_SEQUENCE))
        self.state = tuple(self.state)
        self.freq = next_state_freq
        self.rocof = next_state_rocof
        ####self.freq, self.rocof = self.state

        return self.state, freqs_all_steps, rocofs_all_steps, num_of_violated_freqs

    def step(self, action, collectPlotData):
        done = self.timestep == N_ACTIONS_IN_SEQUENCE - 1
        '''
        if done:
            next_state = self.state #will be set to None in the training loop
            reward = 0.0
            freqs = []
            rocofs = []
        else:
        '''
        ##########self.disturbance = self.disturbance + action
        next_state, freqs_all_steps, rocofs_all_steps, num_of_violated_freqs = self.update_state(action, collectPlotData)
        reward = self.calculate_reward(action, num_of_violated_freqs)

        return next_state, reward, done, freqs_all_steps, rocofs_all_steps


    def calculate_reward(self, action, num_of_violated_freqs):
        reward = 0
        reward = -0.05 * num_of_violated_freqs
        #for one_generator_freq in self.freq:
            #if (one_generator_freq < self.low_freq_limit or one_generator_freq > self.high_freq_limit):
                #reward -= 0.5

        total_control_effort = 0.0
        for vsc_setpoint in action:
            total_control_effort += abs(vsc_setpoint) 
        reward = reward - 0.5 * total_control_effort

        self.action_sum += action

        return reward

    #TODO DISTURBANCE BI TREBALO IZ DATASETA DA SE CITA, A MOZDA I NESTO VISE, TIPA PODACI O TRENUTNOJ POTROSNJI?
    def reset(self, initial_disturbance_dict):
        self.freq = 0 #todo  ovo bi sad trebalo da je lista, ali nije bitno
        self.rocof = 0
        ##########self.disturbance = initial_disturbance
        self.action_sum = [0 for i in range(self.action_space_dims)]
        self.square_root_sum_action_squared = 0
        self.timestep = 0
        self.state = self.scipy_model.reset_model(initial_disturbance_dict)
        self.state.append(self.timestep / float(N_ACTIONS_IN_SEQUENCE))
        self.state = tuple(self.state)

        return self.state


'''
Comment1
u = [p_vsc1, p_vsc2, p_vsc3, load_vsc1, load_vsc2, load_vsc3, p_sm1, …, p_sm7, load_sm1, …, load_sm7, load_bus11, … load_bus39]
nodeid 0         1      2        3           4          5        6    ..   12     13     ..   19          20      ...    48
'''