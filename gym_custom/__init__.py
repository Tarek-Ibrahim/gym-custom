from gym.envs.registration import register

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
    id='halfcheetah_custom-v1',
    entry_point='gym_custom.envs:hce_torch',
    max_episode_steps=200,
    reward_threshold=4800.0,
)

register(
    id='halfcheetah_custom-v2',
    entry_point='gym_custom.envs:hce_tf',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)