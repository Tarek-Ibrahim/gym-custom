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