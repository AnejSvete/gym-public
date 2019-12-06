import math

import numpy as np
from scipy.constants import g, pi

import gym
from gym import spaces
from gym.utils import seeding


class MountainCarExtendedEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):
        self.min_position, self.max_position = -1.2, 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 10 / 10000
        self.gravity = g / 4000

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.state = None
        self.track = None
        self.car_trans, self.pole_trans = None, None
        self.np_random = None

        ######################################################################
        self.mass_car = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_car
        self.length = 0.25  # actually half the pole's length
        self.pole_mass_length = self.mass_pole * self.length

        self.min_theta, self.max_theta = -pi / 4, pi / 4
        self.tau = 0.02  # seconds between state updates
        ######################################################################

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def theta_acc(self, theta, theta_dot, force):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        theta_dot_dot = ((self.gravity * sin_theta * self.total_mass - cos_theta *
                          (force + self.pole_mass_length * theta_dot * theta_dot * sin_theta)) /
                         (self.length * (4.0 / 3.0 * self.total_mass - self.mass_pole * cos_theta * cos_theta)))
        return theta_dot_dot

    def x_acc(self, theta, theta_dot, theta_dot_dot, force):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        x_dot_dot = ((force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass -
                     self.pole_mass_length * theta_dot_dot * cos_theta / self.total_mass)
        return x_dot_dot

    def step(self, action: int):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        x, x_dot, theta, theta_dot = self.state

        ######################################################################
        # force = self.force if action == 1 else -self.force
        force = (action - 1) * 10
        theta_dot_dot = self.theta_acc(theta, theta_dot, force)
        # x_dot_dot = self.x_acc(theta, theta_dot, theta_dot_dot, force)
        # x += + self.tau * x_dot
        # x_dot += + self.tau * x_dot_dot
        theta += + self.tau * theta_dot
        theta_dot += + self.tau * theta_dot_dot
        ######################################################################

        x_dot += (action - 1) * self.force + math.cos(3 * x) * self.total_mass * (-self.gravity)
        x_dot = np.clip(x_dot, -self.max_speed, self.max_speed)
        x += x_dot
        x = np.clip(x, self.min_position, self.max_position)
        if x == self.min_position and x_dot < 0:
            x_dot = 0

        done = x >= self.goal_position and x_dot >= self.goal_velocity
        reward = -1.0

        # theta_dot = np.random.uniform(low=-pi/30, high=pi/30)
        # theta += theta_dot
        # theta = np.clip(theta, self.min_theta, self.max_theta)

        self.state = (x, x_dot, theta, theta_dot)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4),
                               0,
                               self.np_random.uniform(low=-0.05, high=0.05),
                               self.np_random.uniform(low=-0.05, high=0.05)])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        car_width = 40
        car_height = 20

        pole_width = 10.0
        pole_length = scale * 2 * self.length

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10
            l, r, t, b = -car_width / 2, car_width / 2, car_height, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.car_trans = rendering.Transform()
            car.add_attr(self.car_trans)
            self.viewer.add_geom(car)

            front_wheel = rendering.make_circle(car_height / 2.5)
            front_wheel.set_color(.5, .5, .5)
            front_wheel.add_attr(rendering.Transform(translation=(car_width / 4, clearance)))
            front_wheel.add_attr(self.car_trans)
            self.viewer.add_geom(front_wheel)

            back_wheel = rendering.make_circle(car_height / 2.5)
            back_wheel.add_attr(rendering.Transform(translation=(-car_width / 4, clearance)))
            back_wheel.add_attr(self.car_trans)
            back_wheel.set_color(.5, .5, .5)
            self.viewer.add_geom(back_wheel)

            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

            ######################################################################
            l, r, t, b = -pole_width / 2, pole_width / 2, pole_length - pole_width / 2, -pole_width / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_trans = rendering.Transform(translation=(0, clearance))
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.car_trans)
            self.viewer.add_geom(pole)
            ######################################################################

        pos = self.state[0]
        self.car_trans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.car_trans.set_rotation(math.cos(3 * pos))

        theta = self.state[2]
        self.pole_trans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
