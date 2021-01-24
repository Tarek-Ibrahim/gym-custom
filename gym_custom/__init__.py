from gym.envs.registration import register

#=========
#Train envs
#=========

register(
    id='env-v0',
    entry_point='gym_custom.envs:cpe0',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='env-v1',
    entry_point='gym_custom.envs:cpe1',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='env-v2',
    entry_point='gym_custom.envs:cpe2',
    max_episode_steps=500,
    reward_threshold=475.0,
)

# register(
#     id='env-v3',
#     entry_point='gym_custom.envs:cpe3',
#     max_episode_steps=500,
#     reward_threshold=475.0,
# )

#=========
#Test envs
#=========

register(
    id='env_te-v0',
    entry_point='gym_custom.envs:cpe_te0',
    max_episode_steps=1e12,
    # reward_threshold=475.0,
)

register(
    id='env_te-v1',
    entry_point='gym_custom.envs:cpe_te1',
    max_episode_steps=1e12,
    # reward_threshold=475.0,
)

register(
    id='env_te-v2',
    entry_point='gym_custom.envs:cpe_te2',
    max_episode_steps=1e12,
    # reward_threshold=475.0,
)

# register(
#     id='env_te-v3',
#     entry_point='gym_custom.envs:cpe_te3',
#     max_episode_steps=1e12,
#     # reward_threshold=475.0,

# )

# register(
#     id='pid_env_te-v0',
#     entry_point='gym_custom.envs:pid_cpe_te0',
#     max_episode_steps=1e12,
#     # reward_threshold=475.0,
# )

# register(
#     id='env_te-v1',
#     entry_point='gym_custom.envs:pid_cpe_te1',
#     max_episode_steps=1e12,
#     # reward_threshold=475.0,
# )

# register(
#     id='env_te-v2',
#     entry_point='gym_custom.envs:pid_cpe_te2',
#     max_episode_steps=1e12,
#     # reward_threshold=475.0,
# )