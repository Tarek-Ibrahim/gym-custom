from gym.envs.registration import register
from .envs.config import CONFIG_PATH
from.envs.assets import MODEL_PATH
import os

#%%
#===============
# Cartpole
#===============

register(
    id='cartpole_custom-v1',
    entry_point='gym_custom.envs:cpe_torch',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='cartpole_custom-v2',
    entry_point='gym_custom.envs:cpe_tf',
    max_episode_steps=200,
    reward_threshold=195.0,
)


register(
    id='cartpole_custom_rand-v2',
    entry_point='gym_custom.envs:rand_wrapper',
    kwargs={'entry_point': 'gym_custom.envs:cpe_tf',
            'config': os.path.join(CONFIG_PATH, "cartpole","default.json")},
    max_episode_steps=200,
    reward_threshold=195.0,
)


#%%
#===============
# Half-Cheetah
#===============

register(
    id='halfcheetah_custom-v1',
    entry_point='gym_custom.envs:hce_torch',
    max_episode_steps= 200, #1000
    reward_threshold=4800.0,
)

register(
    id='halfcheetah_custom_norm-v1',
    entry_point='gym_custom.envs:norm_wrapper',
    kwargs={'entry_point': 'gym_custom.envs:hce_torch'},
    max_episode_steps= 200, #1000
    reward_threshold=4800.0,
)


register(
    id='halfcheetah_custom_rand-v1',
    entry_point='gym_custom.envs:rand_wrapper',
    kwargs={'entry_point': 'gym_custom.envs:hce_rand',
            'xml_name': os.path.join(MODEL_PATH,"half_cheetah.xml"),
            'config': os.path.join(CONFIG_PATH, "halfcheetah","default.json"),
            'rand': 'size'}, #limbs sizes/lengths to randomzie
    max_episode_steps= 500, #200 #1000
    reward_threshold=4800.0,
)


register(
    id='halfcheetah_custom_rand-v2',
    entry_point='gym_custom.envs:rand_wrapper',
    kwargs={'entry_point': 'gym_custom.envs:hce_rand',
            'xml_name': os.path.join(MODEL_PATH,"half_cheetah.xml"),
            'config': os.path.join(CONFIG_PATH, "halfcheetah","friction.json"),
            'rand': 'friction'},
    max_episode_steps= 500, #200 #1000
    reward_threshold=4800.0,
)


register(
    id='halfcheetah_custom-v2',
    entry_point='gym_custom.envs:hce_tf',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)


#%%
#=========
# Hopper
#=========


register(
    id='hopper_custom_rand-v1',
    entry_point='gym_custom.envs:rand_wrapper',
    kwargs={'entry_point': 'gym_custom.envs:he_rand',
            'xml_name': os.path.join(MODEL_PATH,"hopper.xml"), #'half_cheetah.xml',
            'config': os.path.join(CONFIG_PATH, "hopper","damping.json"),
            'rand': 'damping'}, #floor friction to randomize
    max_episode_steps= 1000,
    reward_threshold=3800.0,
)


register(
    id='hopper_custom_rand-v2',
    entry_point='gym_custom.envs:rand_wrapper',
    kwargs={'entry_point': 'gym_custom.envs:he_rand',
            'xml_name': os.path.join(MODEL_PATH,"hopper.xml"), #'half_cheetah.xml',
            'config': os.path.join(CONFIG_PATH, "hopper","friction.json"),
            'rand': 'friction'}, #floor friction to randomize
    max_episode_steps= 1000,
    reward_threshold=3800.0,
)


#%%
#===============
# Lunar Lander
#===============

register(
    id='lunarlander_custom_default_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\default.json',
            'rand': 'strength'}
)

register(
    id='lunarlander_custom_10_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\10.json'}
)

register(
    id='lunarlander_custom_16_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\16.json'}
)

register(
    id='lunarlander_custom_820_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': os.path.join(CONFIG_PATH, "lunarlander","random_820.json"),
            'rand': 'strength'}
)

register(
    id='lunarlander_custom_2D820_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\random2D_820.json'}
)

register(
    id='lunarlander_custom_811_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\random_811.json'}
)

register(
    id='lunarlander_custom_812_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\random_812.json'}
)

register(
    id='lunarlander_custom_813_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\random_813.json'}
)

register(
    id='lunarlander_custom_1720_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\random_1720.json'}
)

register(
    id='lunarlander_custom_620_rand-v0',
    entry_point='gym_custom.envs:rand_wrapper',
    max_episode_steps=1000,
    kwargs={'entry_point': 'gym_custom.envs:lle_rand',
            'config': CONFIG_PATH + '\\lunarlander\\random_620.json'}
)