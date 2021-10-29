# classical control
# cartpole
from gym_custom.envs.cartpole_custom_torch import CartPoleEnv as cpe_torch
from gym_custom.envs.cartpole_custom_tf import CartPoleEnv as cpe_tf

# mujoco
# half-cheetah
from gym_custom.envs.half_cheetah_custom_torch import HalfCheetahEnv as hce_torch
from gym_custom.envs.half_cheetah_custom_tf import HalfCheetahEnv as hce_tf
from gym_custom.envs.utils import mujoco_wrapper as wrapper