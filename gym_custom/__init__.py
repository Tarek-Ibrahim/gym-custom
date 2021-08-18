from gym.envs.registration import register

register(
    id='cartpole_custom-v1',
    entry_point='gym_custom.envs:cpe',
    max_episode_steps=200,
    reward_threshold=195.0,
)