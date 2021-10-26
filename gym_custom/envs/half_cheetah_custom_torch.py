from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import torch

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle): #(HalfCheetahEnv_):
    def __init__(self):
        self.task = 1.0
        self.prev_qpos = None
        self.action_mask=1.0
        # super().__init__()
        
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)
        
        self.init_geom_rgba=self.model.geom_rgba.copy()
    
    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat) ##
        return self._get_obs()

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
            
        
# %% version 1
    
    def cost_o(self,o):
        return -o[:, 0]
        # return -o[:, 8] - (o[:,1].cos()+o[:,1].sin())

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
        # action=np.clip(action, self.action_space.low, self.action_space.high)
        action=self.action_mask*action
        self.prev_qpos = np.copy(self.sim.data.qpos.flat) #self.sim.data.qpos
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        vel=(self.sim.data.qpos[0] - self.prev_qpos[0]) / self.dt
        reward_ctrl = -0.1 * np.square(action).sum() # np.sum(np.square(action))
        reward_run = ob[0] #self.task * vel #ob[0] 
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat #+3 = center of mass coordinates
        ])#.astype(np.float32).flatten()

    
# %% version 2
    
    # def cost_o(self,o):
    #     # return -o[:, 0] - (o[:,2].cos()+o[:,2].sin())
    #     return -o[:, 8] - (o[:,1].cos()+o[:,1].sin()) 
        
    # def obs_preproc(self, obs):
    #     return obs
        # if isinstance(obs, np.ndarray):
        #     return np.concatenate([obs[:, :2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)
        # elif isinstance(obs, torch.Tensor):
        #     return torch.cat([obs[:, :2],obs[:, 2:3].sin(),obs[:, 2:3].cos(),obs[:, 3:]], dim=1)
    
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