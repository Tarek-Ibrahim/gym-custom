from gym_custom.envs.utils import norm_wrapper, rand_wrapper

# classical control
# cartpole
from gym_custom.envs.cartpole_custom_torch import CartPoleEnv as cpe_torch
from gym_custom.envs.cartpole_custom_tf import CartPoleEnv as cpe_tf

# mujoco
# half-cheetah
from gym_custom.envs.half_cheetah_custom_torch import HalfCheetahEnv as hce_torch
from gym_custom.envs.half_cheetah import HalfCheetahRandomizedEnv as hce_rand
from gym_custom.envs.half_cheetah_custom_tf import HalfCheetahEnv as hce_tf
#hopper
from gym_custom.envs.hopper import HopperRandomizedEnv as he_rand

#box2d
#lunar lander
from gym_custom.envs.lunar_lander import LunarLanderRandomized as lle_rand