import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi

from shapely.geometry import LineString, Polygon


def angle(ba, bc):
    return np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))


class CartPoleHillEquationsEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.world_width, self.world_height = 4 * pi, 2 * pi

        self.gravity = -g
        self.mass_cart = 10.0
        self.mass_pole = 1
        self.length = 0.5
        self.force_mag = 0.0

        self.tau = 0.02

        self.theta_min, self.theta_max = -pi / 12, pi / 12
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([self.x_min * 2, -np.finfo(np.float32).max, self.theta_min * 2, -np.finfo(np.float32).max])
        high = np.array([self.x_max * 2, np.finfo(np.float32).max, self.theta_max * 2, np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.screen_width_pixels, self.screen_height_pixels = 1600, 800
        self.scale = self.screen_width_pixels / self.world_width

        self.pole_width_pixels = 20.0
        self.pole_length_pixels = self.scale * (2 * self.length)

        self.pole_length = 2 * self.length

        self.seed()
        self.viewer = None
        self.previous_state, self.state = None, None

        self.times_at_goal = 0

    def reset(self):
        # self.state = self.np_random.uniform(low=(0.0, 0.0, 0.0, 0.0),
        #                                     high=(0.0, 0.0, 0.0, 0.0),
        #                                     size=(4,))
        self.state = [pi / 2, 0, -pi/4, 0]
        # self.state = [9 * pi/25, 0, 0, 0]
        self.times_at_goal = 0
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def x(self, s):
        return s

    def x_dot(self, s):
        return 1

    def x_dot_dot(self, s):
        return 0

    def y(self, s):
        return [pi] * len(s) if type(s) not in [float, np.float64] else pi

    def y_dot(self, s):
        return 0

    def y_dot_dot(self, s):
        return 0

    def s_dot_dot(self, force=0.0):
        s, s_dot, theta, theta_dot = self.state
        return (-2 * force + self.y_dot(s) * (-(self.gravity * (self.mass_pole + 2 * self.mass_cart +
                                                                self.mass_pole * np.cos(2*theta))) -
                                              2 * self.pole_length * self.mass_pole * np.cos(theta) * theta_dot**2 +
                                              s_dot**2 * (self.mass_pole * np.sin(2 * theta) * self.x_dot_dot(s) +
                                                          (self.mass_pole + 2 * self.mass_cart +
                                                           self.mass_pole * np.cos(2 * theta)) * self.y_dot_dot(s))) +
                self.x_dot(s) * (-(self.gravity * self.mass_pole * np.sin(2*theta)) -
                                 2 * self.pole_length * self.mass_pole * np.sin(theta) * theta_dot**2 +
                                 s_dot**2 * ((self.mass_pole + 2 * self.mass_cart -
                                              self.mass_pole * np.cos(2*theta)) * self.x_dot_dot(s) +
                                             self.mass_pole * np.sin(2 * theta) * self.y_dot_dot(s)))) / \
               ((-self.mass_pole - 2 * self.mass_cart + self.mass_pole * np.cos(2*theta)) * self.x_dot(s)**2 -
                2 * self.mass_pole * np.sin(2 * theta) * self.x_dot(s) * self.y_dot(s) -
                (self.mass_pole + 2*self.mass_cart + self.mass_pole*np.cos(2*theta))*self.y_dot(s)**2)

    def theta_dot_dot(self, force=0.0):
        s, s_dot, theta, theta_dot = self.state
        return (force * (np.cos(theta) * self.x_dot(s) - np.sin(theta) * self.y_dot(s)) +
                (np.sin(theta) * self.x_dot(s) + np.cos(theta) * self.y_dot(s)) *
                (self.y_dot(s) * (-(self.pole_length * self.mass_pole * np.sin(theta) * theta_dot**2) +
                                  (self.mass_pole + self.mass_cart) * s_dot**2 * self.x_dot_dot(s)) +
                 self.x_dot(s) * (self.pole_length * self.mass_pole * np.cos(theta) * theta_dot**2 +
                                  (self.mass_pole + self.mass_cart) *
                                  (self.gravity - s_dot**2 * self.y_dot_dot(s))))) / \
               (self.pole_length * ((-self.mass_pole - self.mass_cart +
                                     self.mass_pole * np.cos(theta)**2) * self.x_dot(s)**2 -
                                    self.mass_pole * np.sin(2 * theta) * self.x_dot(s) * self.y_dot(s) -
                                    ((self.mass_pole + 2 * self.mass_cart +
                                      self.mass_pole * np.cos(2 * theta)) * self.y_dot(s)**2) / 2))

    def new_state(self, action):

        s, s_dot, theta, theta_dot = self.state
        force = self.force_mag * (int(action) - 1)
        # print([s, s_dot, theta, theta_dot])

        s_dot_dot = self.s_dot_dot(force)
        theta_dot_dot = self.theta_dot_dot(force)

        s += self.tau * s_dot
        s_dot += self.tau * s_dot_dot
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_dot_dot

        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        return np.array([s, s_dot, theta, theta_dot])

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.state = self.new_state(action)

        return np.array(self.state), 0, False, {}

    def render(self, mode='human'):

        s, s_dot, theta, theta_dot = self.state

        x = self.x(s)

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width_pixels, self.screen_height_pixels)

            # track / ground
            xs = np.linspace(self.x_min, self.x_max, 2000)
            ys = np.array(self.y(xs))
            xys = list(zip((xs - self.x_min) * self.scale, ys * self.scale))

            self.track = rendering.make_polyline([(0, 0), *xys, (self.screen_width_pixels, 0)])
            self.track.set_linewidth(5)
            self.viewer.add_geom(self.track)

            # cart
            cart = rendering.make_circle(self.pole_width_pixels)
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # pole
            pole_line = LineString([(0, 0), (0, self.pole_length_pixels)]).buffer(self.pole_width_pixels / 2)
            pole = rendering.make_polygon(list(pole_line.exterior.coords))
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_trans = rendering.Transform()
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

        self.cart_trans.set_translation((x - self.x_min) * self.scale, self.y(x) * self.scale)
        k = np.arctan(-1 / self.y_dot(x)) if self.y_dot(x) != 0.0 else pi / 2
        self.cart_trans.set_rotation(pi / 2 + k if k < 0 else k - pi / 2)

        self.pole_trans.set_rotation(-(pi / 2 + k if k < 0 else k - pi / 2) - theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
