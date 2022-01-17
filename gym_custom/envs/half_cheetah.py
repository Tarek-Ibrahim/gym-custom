import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from .randomized_locomotion import RandomizedLocomotionEnv

class HalfCheetahRandomizedEnv(RandomizedLocomotionEnv):
    def __init__(self, **kwargs):
        self.task = 1.0
        self.prev_qpos = None
        self.action_mask=1.0
        
        RandomizedLocomotionEnv.__init__(self, **kwargs)
    
    def sample_tasks(self, num_tasks): #gives -1 (backward) or +1 (forward) with 50% probability #manually designed p(T) #goal: going in different directions
        directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        return directions
    
    def sample_task(self): #samples joint idx to be disabled
        return np.random.randint(0, self.action_space.shape[0])
    
    def reset_task(self,task,task_name=None):
        
        #environment-based
        if task_name=="cripple":
            self.action_mask=np.ones(self.action_space.shape)
            self.action_mask[task]=0. #1.
            
            #disabled joint visualization
            geom_idx = self.model.geom_names.index(self.model.joint_names[task+3])
            if 'thigh' in self.model.joint_names[task+3]:
                other=self.model.geom_names.index('torso')
            else:
                other=self.model.geom_names.index(self.model.joint_names[task+2])
            geom_rgba = self.init_geom_rgba.copy()
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
            geom_rgba[other, :3] = np.array([1, 0, 0])
            self.model.geom_rgba[:] = geom_rgba
            
        else: #reward-based
            self.task=task
    
    def step(self, action):
        # action=np.clip(action, self.action_space.low, self.action_space.high)
        action=self.action_mask*action
        self.prev_qpos = np.copy(self.sim.data.qpos.flat) #self.sim.data.qpos
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        vel=(self.sim.data.qpos[0] - self.prev_qpos[0]) / self.dt
        reward_ctrl = -0.05 * np.square(action).sum() # np.sum(np.square(action))
        reward_run = self.task * vel #ob[0] 
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}
    
    def _get_obs(self):
        return np.concatenate([
            # (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat #+3 = center of mass coordinates
        ])#.astype(np.float32).flatten()

    # def step(self, action):
    #     xposbefore = self.sim.data.qpos[0]
    #     self.do_simulation(action, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     ob = self._get_obs()
    #     reward_ctrl = - 0.1 * np.square(action).sum()
    #     reward_run = (xposafter - xposbefore)/self.dt
    #     reward = reward_ctrl + reward_run
    #     done = False
    #     return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[1:],
    #         self.sim.data.qvel.flat,
    #     ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
