import gym
from gym import spaces
import random
import numpy as np
from scipy import signal
import pickle
from gym.spaces import Tuple
from gym.spaces.space import Space
from config import *

#referent directions of active powers:
#negative disturbance - demand increase (frequency decrease)
#positive action - 'generation' increase

#Custom space
class Incremental(Space):
    def __init__(self, start, stop, step, **kwargs):
        self.step = step
        self.size = int((stop - start) / step) + 1
        self.values = np.linspace(start, stop, self.size, **kwargs)
        super().__init__(self.values.shape, self.values.dtype)

    def sample(self):
        return np.random.choice(self.values)

    def contains(self, x):
        return x in self.values
        
class ScipyModel():
    def __init__(self):
        # Define the frequnecy response state-space
        A = [[0,1],[-1.629351428850842,-1.911954183169527]]
        B = [[0],[1]]
        C = [3.745887382384376,7.491774764768751]
        D = [0]
        sys = signal.StateSpace(A,B,C,D)

        # Define simulation parameters

        self.nSteps = 175 # Total number of time steps
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
        self.x = np.zeros((2,self.nSteps))
        self.f = np.zeros((1,self.nSteps))
        self.rf = np.zeros((1,self.nSteps))
        self.u = np.zeros((1,self.nSteps)) #disturbance
        self.t = np.zeros((1,self.nSteps))

    def reset_model(self, initial_disturbance):
        self.x = np.zeros((2,self.nSteps+2))
        self.f = np.zeros((1,self.nSteps))
        self.rf = np.zeros((1,self.nSteps))
        self.u = np.zeros((1,self.nSteps+1))
        self.u[:,self.nDist:self.nSteps] = initial_disturbance
        self.t = np.zeros((1,self.nSteps))

        for i in range (self.n_agent_timestep_steps):
            self.x[:,i+1] = np.dot(self.Ad, self.x[:,i]) + np.dot(self.Bd, self.u[:,i])
            self.f[:,i] = np.dot(self.Cd, self.x[:,i]) + np.dot(self.Dd, self.u[:,i]) + 50.0
            if i != 0:
                self.rf[:,i] = (self.f[:,i] - self.f[:,i-1])/self.Td
            self.t[:,i] = i*self.Td 
        
        #test
        if (i*self.Td != 0.24):
            print('WARNING in environment_discrete.py: i*self.Td != 0.24')

        #todo check if this is ok when more resources are added
        state = []
        state += self.f[:,i].tolist()
        state += self.rf[:,i].tolist()

        return state
            
    #agent timestep count starts from zero, but here, during the reset_model, 
    #the amount of n_agent_timestep_steps has been done
    def get_next_state(self, agent_timestep, action, collectPlotData):
        current_step = self.n_agent_timestep_steps * (agent_timestep + 1)
        for i in range(current_step, current_step + self.n_agent_timestep_steps):
            self.x[:,i+1] = np.dot(self.Ad, self.x[:,i]) + np.dot(self.Bd, self.u[:,i])
            self.f[:,i] = np.dot(self.Cd, self.x[:,i]) + np.dot(self.Dd, self.u[:,i]) + 50.0
            if i != 0:
                self.rf[:,i] = (self.f[:,i] - self.f[:,i-1])/self.Td
            self.t[:,i] = i*self.Td 
            
            self.u[:,i+1] = self.u[:,current_step - 1] + action

        # todo check if this is ok when more resources are added
        next_state = []
        next_state += self.f[:,i].tolist()
        next_state += self.rf[:,i].tolist()

        # todo bice problema kada budemo imali vise resursa.
        # najbolje tada da saljemo listu listi frekvencija, pa da je u okviru test metode raspakujemo
        # iteriraj po numpy redovima i svaki red konvertuj u listu, pa salji listu listi 
        if (collectPlotData and i == self.nSteps - 1):
            freqs = self.f.tolist()[0]
            rocofs = self.rf.tolist()[0]
            #print(self.u)
        else:
            freqs = []
            rocofs = []

        return next_state, freqs, rocofs


class EnvironmentDiscrete(gym.Env):
    
    def __init__(self):
        super(EnvironmentDiscrete, self).__init__()
        
        self.freq = 50
        self.rocof = 0
        self.timestep = 0
        self.state = (self.freq, self.rocof, self.timestep * 1.0 )
        self.disturbance = 0
        self.action_sum = 0 #models setpoint change, that should be zero at the end

        self.min_disturbance = -0.1
        self.max_disturbance = 0.0

        self.state_space_dims = 3 #f i rocof i timestep
        self.action_space_dims = 1 #delta P

        self.low_set_point = -0.1
        self.high_set_point = 0.1
        self.action_space = Incremental(self.low_set_point, self.high_set_point, 0.01)
        self.n_actions = self.action_space.size

        #self.regression_model = pickle.load(open('regression.sav', 'rb'))
        self.scipy_model = ScipyModel()

        self.low_freq_limit = 49.85
        self.high_freq_limit = 50.0

    def update_state(self, action, collectPlotData):
        #features = np.array([[self.disturbance, self.freq, self.rocof]])
        #next_state = self.regression_model.predict(features)[0] #list

        next_state, freqs, rocofs = self.scipy_model.get_next_state(self.timestep, action, collectPlotData)

        self.state = next_state
        self.state.append(self.timestep * 1.0)
        self.state = tuple(self.state)
        self.freq, self.rocof, _ = self.state

        return self.state, freqs, rocofs

    def step(self, action, collectPlotData):
        done = self.timestep == N_ACTIONS_IN_SEQUENCE
        if done:
            next_state = self.state #will be set to None in the training loop
            reward = 0.0
            freqs = []
            rocofs = []
        else:
            self.disturbance = self.disturbance + action
            next_state, freqs, rocofs = self.update_state(action, collectPlotData)
            reward = self.calculate_reward(action)
            self.timestep += 1

        return next_state, reward, done, freqs, rocofs


    def calculate_reward(self, action):
        reward = 0
        if (self.freq < self.low_freq_limit or self.freq > self.high_freq_limit):
            reward -= 100.0
        reward = reward - 1.0 * abs(action) #control effort

        self.action_sum += action
        #todo provjeri jos jednom je li ok ovaj predzanji trenutak
        if self.timestep == N_ACTIONS_IN_SEQUENCE - 1:
            reward -= 100.0 * abs(self.action_sum)

        return reward

    #TODO DISTURBANCE BI TREBALO IZ DATASETA DA SE CITA, A MOZDA I NESTO VISE, TIPA PODACI O TRENUTNOJ POTROSNJI?
    def reset(self, initial_disturbance):
        self.freq = 50
        self.rocof = 0
        self.disturbance = initial_disturbance
        self.action_sum = 0
        self.timestep = 0
        self.state = self.scipy_model.reset_model(initial_disturbance)
        self.state.append(self.timestep * 1.0)
        self.state = tuple(self.state)

        return self.state
