"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from copy import copy, deepcopy


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.force=0.0
        self.smooth=0.9 #0.75
        self.enforce_smooth=True
        # self.saturate=True
        self.T=0.0
        self.td=0.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

#         self.action_space = spaces.Discrete(201) #range(21) #np.arange(-1,1,0.01) #spaces.Discrete(2) #np.arange(-1,1,0.01)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
        
        self.seed()
        self.eps = np.finfo(np.float32).eps.item()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
#         assert self.action_space.contains(action), err_msg

        # act=np.sign(action[0]) if np.abs(action[0])>1. else copy(action[0])

        x, x_dot, theta, theta_dot = self.state
        
        #calculate force

        # force=act*self.force_mag
        t_limit_done=False
        if action[0] > 1e5: t_limit_done=True
        action=np.clip(action[0], -1.0, 1.0)
        force = action * self.force_mag
        delta=np.abs(self.force-force)#/self.tau

        if self.enforce_smooth and np.abs(self.force-force)>=1: force=(1-self.smooth)*self.force+self.smooth*force

        
        # if self.enforce_smooth: 
        #     if np.abs(self.force-force)>=1: #ensure continuity
        #         self.smooth=np.abs(1/((self.force-force)+self.eps))
        #         force=(1-self.smooth)*self.force+self.smooth*force
            
        # if self.saturate: force= np.sign(force)*self.force_mag if np.abs(force)>self.force_mag else force #saturate

        self.force=deepcopy(force) #store prev force
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or t_limit_done
        )

        diff=np.abs(theta_dot-self.td)*(180/np.pi)
        self.td=theta_dot
        theta_deg=theta*(180./np.pi)
        theta_thr_deg=self.theta_threshold_radians*(180./np.pi)
        #baselines 1: a=0.05; b=f=g=e=0; c=1; d=1; no smooth (in 0, 3 & te_3)
        #baselines 2: a=0.05; b=f=e=0; g=0.0001; c=1; d=1; no smooth
        #baselines 3: a=0.05; b=f=e=g=0; h=2.0; c=1; d=1.5; no smooth
        #baselines 4 (?): a=0.05; b=f=e=g=0; e=2.0; c=1; d=2.0; no smooth
        #baselines 6: a=0.05; b=100.0; c=1.0; d=0.25; rest=0; smooth const 0.75 [X]
        a=0.05
        b=100.0 #0.2
        c=1.
        d=.25 #1.0 #0.2
        e=0.0 #18.0#*self.T #0.72
        f=0.0 #0.0001 #0.1
        g=0.000#1
        h=0.0 #3.0
        P=np.array([
            [-a,0.,0.,0.],
            [0.,-b,0.,0.],
            [0.,0.,c,0.],
            [0.,0.,0.,-d]
            ])
        ds=copy(np.array(self.state))
        ds[2]=np.abs(theta_deg)-theta_thr_deg #np.abs(theta)-self.theta_threshold_radian
        ds[3]=theta_dot*(180./np.pi)
        # thr=np.array([self.x_threshold,theta_thr_deg])
        # real=np.array([x,theta_deg])
        self.T+=1
        if not done:
            # reward= 0.5*((np.abs(real)-thr)@(np.abs(real)-thr).T) - delta**2 - 2*np.abs(force)**2
            # reward = 0.5*(np.abs(theta_deg)-theta_thr_deg)**2 - 3*delta**2 - 4*np.abs(force)**2 #+ 1/(delta+self.eps) #+ 1/(np.abs(force)+self.eps) #1.0
            # reward=10-(1-math.cos(np.abs(theta)))-5e-3*force**2
            # reward=0.5*(np.abs(theta_deg)-theta_thr_deg)**2 -.072*(theta_dot*(180./np.pi))**2-.0072*force**2
            reward=0.5*ds@P@ds.T-e*force**2-f*delta**2-g*thetaacc**2-h*diff**2
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            # reward = 0.5*(np.abs(theta_deg)-theta_thr_deg)**2 -.072*(theta_dot*(180./np.pi))**2-.0072*force**2
            if (np.abs(x)>=self.x_threshold): P[0,0]=-100 #-30*a
            reward=0.5*ds@P@ds.T-e*force**2-f*delta**2-g*thetaacc**2-h*diff**2
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        
        reward=0 if reward<0. else reward
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.force=0.0
        self.T=0.0
        self.td=0.0
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None