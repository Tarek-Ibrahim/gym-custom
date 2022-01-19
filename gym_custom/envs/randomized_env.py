import gym
import json
import numpy as np

class Dimension(object):
    """Class which handles the machinery for doing BO over a particular dimensions
    """
    def __init__(self, default_value, multiplier_min=0.0, multiplier_max=1.0, name=None):
        """Generates datapoints at specified discretization, and initializes BO
        """
        self.default_value = default_value
        self.current_value = default_value
        self.multiplier_min = multiplier_min 
        self.multiplier_max = multiplier_max
        self.range_min = self.default_value * self.multiplier_min
        self.range_max = self.default_value * self.multiplier_max
        self.name = name

    def _rescale(self, value):
        """Rescales normalized value to be within range of env. dimension"""
        return self.range_min + (self.range_max - self.range_min) * value

    def randomize(self):
        self.current_value = np.random.uniform(low=self.range_min, high=self.range_max)

    def reset(self):
        self.current_value = self.default_value

    def set(self, value):
        self.current_value = value 



class RandomizedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RandomizedEnvWrapper, self).__init__(env)
        self.config_file = self.unwrapped.config_file

        self._load_randomization_dimensions()
        self.unwrapped._update_randomized_params()

    def _load_randomization_dimensions(self):
        """load environment defaults ranges"""
        self.unwrapped.dimensions = []

        with open(self.config_file, mode='r') as f:
            config = json.load(f)

        for dimension in config['dimensions']:
            self.unwrapped.dimensions.append(
                Dimension(
                    default_value=dimension['default'],
                    multiplier_min=dimension['multiplier_min'],
                    multiplier_max=dimension['multiplier_max'],
                    name=dimension['name']
                )
            )

        nrand = len(self.unwrapped.dimensions)
        self.unwrapped.randomization_space = gym.spaces.Box(0, 1, shape=(nrand,), dtype=np.float32)


    def randomize(self, randomized_values=-1):
        """Creates a randomized environment, using the dimension and value specified 
        to randomize over
        """
        for dimension, randomized_value in enumerate(randomized_values):
            if randomized_value == 'default':
                self.unwrapped.dimensions[dimension].current_value = self.unwrapped.dimensions[dimension].default_value
            elif randomized_value != 'random' and randomized_value != -1:
                assert 0.0 <= randomized_value <= 1.0, "using incorrect: {}".format(randomized_value)
                self.unwrapped.dimensions[dimension].current_value = self.unwrapped.dimensions[dimension]._rescale(randomized_value)
            else:  # random
                self.unwrapped.dimensions[dimension].randomize()

        self.unwrapped._update_randomized_params()
        
        # return self.unwrapped.dimensions[0].current_value

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)