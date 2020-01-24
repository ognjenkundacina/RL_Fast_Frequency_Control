import gym
from gym import spaces
import random
import numpy as np
import pickle
from gym.spaces import Tuple
from gym.spaces.space import Space

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

class EnvironmentDiscrete(gym.Env):
    
    def __init__(self):
        super(EnvironmentDiscrete, self).__init__()
        
        self.freq = 50
        self.rocof = 0
        self.state = (self.freq, self.rocof)
        self.disturbance = 0

        self.min_disturbance = -0.1
        self.max_disturbance = 0.1

        self.state_space_dims = 2 #f i rocof
        self.action_space_dims = 1 #delta P

        self.low_set_point = -0.1
        self.high_set_point = 0.1
        self.action_space = Incremental(self.low_set_point, self.high_set_point, 0.01)
        self.n_actions = self.action_space.size

        self.timestep = 0
        self.regression_model = pickle.load(open('regression.sav', 'rb'))

        self.low_freq_limit = 49.9
        self.high_freq_limit = 50.1

    def update_state(self, action):
        features = np.array([[self.disturbance, self.freq, self.rocof]])
        next_state = self.regression_model.predict(features)[0] #list
        self.state = tuple(next_state)
        self.freq, self.rocof = self.state

        return self.state

    #action: 0..n_actions
    def step(self, action):
        self.disturbance = self.disturbance + action
        next_state = self.update_state(action)
        reward = self.calculate_reward(action)
        done = self.timestep == 4 #PROVJERI
        self.timestep += 1

        return next_state, reward, done


    def calculate_reward(self, action):
        reward = 0
        if (self.freq < self.low_freq_limit or self.freq > self.high_freq_limit):
            reward -= 10.0
        reward = reward - abs(action) #control effort

        #todo = da li zelimo da samo jednom ii dvaput to citavom horizontu agent donosi akcije
        # da li zelimo i KAZNU ZA ROCOF - moze se desiti da on natjera frekvenciju da ne predje granice do zadnjeg timestempa, a da nagib bude ogroman
        return reward

    #TODO DISTURBANCE BI TREBALO IZ DATASETA DA SE CITA, A MOZDA I NESTO VISE, TIPA PODACI O TRENUTNOJ POTROSNJI?
    def reset(self, disturbance):
        self.freq = 50
        self.rocof = 0
        self.state = (self.freq, self.rocof)
        self.disturbance = disturbance
        self.timestep = 0

        return self.state
