from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import torch

class HalfCheetahEnv(HalfCheetahEnv_): #(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.task = 1.0
        self.prev_qpos = None
        super().__init__()
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        # utils.EzPickle.__init__(self)
    
    # def reset_model(self):
    #     qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
    #     qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
    #     self.set_state(qpos, qvel)
    #     self.prev_qpos = np.copy(self.sim.data.qpos.flat) ##
    #     return self._get_obs()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35 #0.5 #0.25
        # self.viewer.cam.elevation = -55
        self.viewer._hide_overlay = True
        
    def cost_a(self,a):
        return 0.1 * (a**2).sum(dim=1)
    
    def sample_tasks(self, num_tasks): #gives -1 (backward) or +1 (forward) with 50% probability #manually designed p(T) #goal: going in different directions
        directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        return directions
    
    def reset_task(self,task):
        self.task=task
        
# %% version 1
    
    def cost_o(self,o):
        return -o[:, 0]

    def obs_preproc(self,obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1) #???: why not feed first item to network?
        elif isinstance(obs, torch.Tensor):
            return torch.cat([obs[:, 1:2],obs[:, 2:3].sin(),obs[:, 2:3].cos(),obs[:, 3:]], dim=1)


    def obs_postproc(self,obs, pred):
        return torch.cat([pred[:, :1], obs[:, 1:] + pred[:, 1:]], dim=1) 


    def targ_proc(self,obs, next_obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1) #!!!: we are predicting first item (at t+1) even tho we are not feeding it as input to the network (at t) [i.e. we are predicting s' w/o s] AND we are predicting it as s' instead of delta_s' like rest of target elements AND we are trying to predict a modification of the first state (rootx) [whereas with the rest it is the state directly]
        elif isinstance(obs, torch.Tensor):
            return torch.cat([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], dim=1)


    def step(self, action):
        self.prev_qpos = self.sim.data.qpos
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        vel=(self.sim.data.qpos[0] - self.prev_qpos[0]) / self.dt
        reward_ctrl = -0.1 * np.sum(np.square(action))
        reward_run = self.task * vel
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            #(self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat #+3 = center of mass coordinates
        ]).astype(np.float32).flatten()

    
# %% version 2
    
    # def cost_o(self,o):
    #     # return -o[:, 0] - (o[:,2].cos()+o[:,2].sin())
    #     return -o[:, 8] - (o[:,1].cos()+o[:,1].sin()) 
        
    # def obs_preproc(self, obs):
    #     return obs
    #     # if isinstance(obs, np.ndarray):
    #     #     return np.concatenate([obs[:, :2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)
    #     # elif isinstance(obs, torch.Tensor):
    #     #     return torch.cat([obs[:, :2],obs[:, 2:3].sin(),obs[:, 2:3].cos(),obs[:, 3:]], dim=1)
    
    # def obs_postproc(self, obs, pred):
    #     return obs + pred 

    # def targ_proc(self, obs, next_obs):
    #     return next_obs - obs
        

    # def _get_obs(self):
    #     return np.concatenate(
    #         [
    #             # (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt, ##
    #             self.sim.data.qpos.flat[1:],
    #             self.sim.data.qvel.flat,
    #         ]
    #     )