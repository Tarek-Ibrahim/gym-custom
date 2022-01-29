from gym.envs.registration import load
from .normalized_env import NormalizedActionWrapper
from .randomized_env import RandomizedEnvWrapper

def norm_wrapper(entry_point, **kwargs):

	# Load the environment from its entry point
	env_cls = load(entry_point)
	env = env_cls(**kwargs)
	# Normalization wrapper
	env = NormalizedActionWrapper(env)

	return env


def rand_wrapper(entry_point, **kwargs):

	# Load the environment from its entry point
	env_cls = load(entry_point)
	env = env_cls(**kwargs)
	# Randomization wrapper
	env = NormalizedActionWrapper(RandomizedEnvWrapper(env))

	return env

