import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi

from shapely.geometry import LineString, Polygon


def angle(ba, bc):
    return np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))


class CartPoleHillExtendedEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.world_width, self.world_height = 4.8, 2.4

        self.gravity = g
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = 0.5  # actually half the pole's length
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_min, self.theta_max = -pi / 12, pi / 12
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([self.x_min * 2, -np.finfo(np.float32).max, self.theta_min * 2, -np.finfo(np.float32).max])
        high = np.array([self.x_max * 2, np.finfo(np.float32).max, self.theta_max * 2, np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.screen_width_pixels, self.screen_height_pixels = 1600, 800
        self.scale = self.screen_width_pixels / self.world_width

        self.cart_y_pixels = 100
        self.cart_width_pixels = 100.0
        self.cart_height_pixels = 60.0
        self.wheels_radius = self.cart_height_pixels / 3

        self.cart_y = self.cart_y_pixels / self.scale
        self.cart_width = self.cart_width_pixels / self.scale
        self.cart_height = self.cart_height_pixels / self.scale

        self.pole_width_pixels = 20.0
        self.pole_length_pixels = self.scale * (2 * self.length)

        self.pole_length = self.pole_length_pixels / self.scale

        self.steepness = 0.5
        self.initial_height = 0.75

        self.goal_position = 2.0

        self.intersection_polygon = None

        self.max_speed = 2.5

        self.seed()
        self.viewer = None
        self.previous_state, self.state = None, None

        self.times_at_goal = 0

    def reset(self):
        self.state = self.np_random.uniform(low=(-2.0 + 1.5, -0.05, -0.05, -0.05),
                                            high=(-1.75 + 1.5, 0.05, 0.05, 0.05),
                                            size=(4,))
        self.times_at_goal = 0
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def height(self, t):

        # return 0.45 * np.sin(2 * (np.clip(t, -pi / 2, pi / 2) - pi / 4)) + 0.75  # !!! FOR TESTING EQUATIONS

        if np.abs(t) <= pi / 8:
            return self.initial_height - self.steepness
        elif np.abs(t) >= pi / 2:
            return self.initial_height + self.steepness
        elif t > 0:
            return self.steepness * np.sin(8/3 * (t + 7 * pi / 16)) + self.initial_height
            # return self.steepness * np.sin(8/3 * (t - 9 * pi / 16)) + self.initial_height
        else:
            return self.steepness * np.sin(8/3 * (t - pi / 16)) + self.initial_height


    def dheight(self, t):

        # return 2 * self.steepness * np.cos(2 * (np.clip(t, -pi / 2, pi / 2) - pi / 4))

        if not pi / 8 <= np.abs(t) <= pi / 2:
            return 0
        elif t > 0:
            return 8/3 * self.steepness * np.cos(8/3 * (t + 7 * pi / 16))
        else:
            return 8/3 * self.steepness * np.cos(8/3 * (t - pi / 16))

    def phi(self, t):
        return np.arctan(self.dheight(t))

    def theta_acc(self, force):
        x, x_dot, theta, theta_dot = self.state
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        theta_dot_dot = ((self.gravity * sin_theta - cos_theta *
                          (force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass) /
                         (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)))
        return theta_dot_dot

    def x_acc(self):
        x, x_dot, theta, theta_dot = self.state
        x_dot_dot = -g * np.sin(self.phi(x))
        return x_dot_dot

    def tmp_x_acc(self, theta_dot_dot, force):
        x, x_dot, theta, theta_dot = self.state
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_dot_dot = ((force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass -
                     self.pole_mass_length * theta_dot_dot * cos_theta / self.total_mass - g * np.sin(self.phi(x)))
        return x_dot_dot

    def new_state(self, action):

        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag * (int(action) - 1)

        theta_dot_dot = self.theta_acc(force)
        x_dot_dot = self.tmp_x_acc(theta_dot_dot, force)
        # x_dot_dot = self.x_acc()

        x += self.tau * x_dot
        x_dot += self.tau * x_dot_dot
        x_dot = np.clip(x_dot, -self.max_speed, self.max_speed)
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_dot_dot

        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        return np.array([x, x_dot, theta, theta_dot])

        # return np.array([x + 0.001, x_dot, theta, theta_dot])

    def reward(self, done):
        current_x, _, _, _ = self.state
        current_distance_from_goal = np.abs(current_x - self.goal_position)
        # return self.times_at_goal if current_distance_from_goal < 0.1 * self.world_width else -1 if done else 0.0
        return 1 / (current_distance_from_goal + 1)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        tmp_state = self.state
        x, x_dot, theta, theta_dot = self.new_state(action)
        self.state = (x, x_dot, theta, theta_dot)
        self.previous_state = tmp_state

        distance_from_goal = np.abs(x - self.goal_position)

        done = not self.x_min <= x <= self.x_max or not self.theta_min <= theta <= self.theta_max or self.times_at_goal >= 50

        if distance_from_goal < 0.1 * self.world_width:
            self.times_at_goal += 1
        else:
            self.times_at_goal = 0

        reward = self.reward(done)

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):

        x, x_dot, theta, theta_dot = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width_pixels, self.screen_height_pixels)

            # track / ground
            xs = np.linspace(self.x_min, self.x_max, 2000)
            # ys = np.array(self.height(xs))
            ys = np.array([self.height(t) for t in xs])
            xys = list(zip((xs - self.x_min) * self.scale, ys * self.scale))

            self.track = rendering.make_polyline([(0, 0), *xys, (self.screen_width_pixels, 0)])
            self.track.set_linewidth(5)
            self.viewer.add_geom(self.track)

            # cart
            l, r, t, b = -self.cart_width_pixels / 2, self.cart_width_pixels / 2, self.cart_height_pixels / 2, -self.cart_height_pixels / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(rendering.Transform(translation=(0, self.wheels_radius + self.cart_height_pixels / 2)))
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # wheels
            front_wheel = rendering.make_circle(self.wheels_radius)
            front_wheel.set_color(0.5, 0.5, 0.5)
            front_wheel.add_attr(rendering.Transform(translation=(self.cart_width_pixels / 4, self.wheels_radius)))
            front_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(front_wheel)
            back_wheel = rendering.make_circle(self.cart_height_pixels / 3)
            back_wheel.set_color(0.5, 0.5, 0.5)
            back_wheel.add_attr(rendering.Transform(translation=(-self.cart_width_pixels / 4, self.wheels_radius)))
            back_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(back_wheel)

            # pole
            pole_line = LineString([(0, 0), (0, self.pole_length_pixels)]).buffer(self.pole_width_pixels / 2)
            pole = rendering.make_polygon(list(pole_line.exterior.coords))
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_trans = rendering.Transform(translation=(0, self.cart_height_pixels + self.wheels_radius))
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

            # axle
            self.axle = rendering.make_circle(self.pole_width_pixels / 2)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            # flag
            flag_x = (self.goal_position - self.x_min) * self.scale
            flag_bottom_y = self.height(self.goal_position) * self.scale
            flag_top_y = flag_bottom_y + 100.0
            flagpole = rendering.Line((flag_x, flag_bottom_y), (flag_x, flag_top_y))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flag_x, flag_top_y), (flag_x, flag_top_y - 25), (flag_x + 50, flag_top_y - 15)])
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        self.cart_trans.set_translation((x - self.x_min) * self.scale, self.height(x) * self.scale)
        k = np.arctan(-1 / self.dheight(x)) if self.dheight(x) != 0.0 else pi / 2
        self.cart_trans.set_rotation(pi / 2 + k if k < 0 else k - pi / 2)

        self.pole_trans.set_rotation(-(pi / 2 + k if k < 0 else k - pi / 2) - theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
