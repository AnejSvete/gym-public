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

        self.theta_min, self.theta_max = -36 * 2 * pi / 360, 36 * 2 * pi / 360
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([self.x_min * 2, -np.finfo(np.float32).max, self.theta_min * 2, -np.finfo(np.float32).max])
        high = np.array([self.x_max * 2, np.finfo(np.float32).max, self.theta_max * 2, np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.screen_width, self.screen_height = 1600, 800
        self.scale = self.screen_width / self.world_width

        self.cart_y = 100
        self.cart_width = 100.0
        self.cart_height = 60.0
        self.track_height = self.cart_y - self.cart_height / 2

        self.cart_y_world = self.cart_y / self.scale
        self.cart_width_world = self.cart_width / self.scale
        self.cart_height_world = self.cart_height / self.scale
        self.track_height_world = self.track_height / self.scale

        self.pole_width = 20.0
        self.pole_length = self.scale * (2 * self.length)
        self.obstacle_width, self.obstacle_height = self.screen_width / 5, self.screen_height / 5  # TODO: convert to world coordinates
        self.obstacle_coordinates = [self.screen_width / 2 - self.obstacle_width / 2,
                                     self.screen_width / 2 + self.obstacle_width / 2,
                                     self.track_height + self.cart_height + self.pole_length + self.obstacle_height / 2 + 150,
                                     self.track_height + self.cart_height + self.pole_length - self.obstacle_height / 2 + 150]

        self.pole_length_world = self.pole_length / self.scale

        self.goal_position_world = 1.9

        self.intersection_polygon = None

        self.max_speed = 1.0

        self.seed()
        self.viewer = None
        self.previous_state, self.state = None, None

        self.already_done = False

    def reset(self):
        self.state = self.np_random.uniform(low=(-2.0 + 1.5, -0.05, -0.05, -0.05),
                                            high=(-1.75 + 1.5, 0.05, 0.05, 0.05),
                                            size=(4,))
        self.already_done = False
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def height(self, t):
        return 0.65 * np.sin(2 * (t - pi / 4)) + 0.75

    def dheight(self, t):
        return 2 * 0.65 * np.cos(2 * (t - pi / 4))

    def theta_acc(self, force):
        x, x_dot, theta, theta_dot = self.state
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        theta_dot_dot = ((self.gravity * sin_theta - cos_theta *
                          (force * self.dheight(x) + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass) /
                         (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)))
        return theta_dot_dot

    def x_acc(self, theta_dot_dot, force):
        x, x_dot, theta, theta_dot = self.state
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_dot_dot = ((force * self.dheight(x) + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass -
                     self.pole_mass_length * theta_dot_dot * cos_theta / self.total_mass - g * self.dheight(x))
        return x_dot_dot

    def pole_top_coordinates(self, screen_coordinates=True, provided_theta=None):
        x, x_dot, theta, theta_dot = self.state
        theta = theta if provided_theta is None else provided_theta
        if screen_coordinates:
            return (x * self.scale + self.screen_width / 2 + self.pole_length * np.sin(theta),
                    self.track_height + self.cart_height + self.pole_length * np.cos(theta))
        else:
            return (x + self.pole_length_world * np.sin(theta),
                    self.track_height_world + self.cart_height_world + self.pole_length_world * np.cos(theta))

    def pole_bottom_coordinates(self, screen_coordinates=True):
        x, x_dot, theta, theta_dot = self.state
        if screen_coordinates:
            return x * self.scale + self.screen_width / 2, self.track_height + self.cart_height
        else:
            return x, self.track_height_world + self.cart_height_world

    def new_state(self, action):

        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag * (int(action) - 1) * 0.01
        theta_dot_dot = self.theta_acc(force)

        x_dot_dot = self.x_acc(theta_dot_dot, force)

        x += self.tau * x_dot
        x_dot += self.tau * x_dot_dot
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_dot_dot

        if theta < -pi:
            theta += 2 * pi
        elif theta > pi:
            theta -= 2 * pi

        return np.array([x, x_dot, theta, theta_dot])

    def reward(self):
        current_x, _, _, _ = self.state
        current_distance_from_goal = np.abs(current_x - self.goal_position_world)
        previous_x, _, _, _ = self.previous_state
        previous_distance_from_goal = np.abs(previous_x - self.goal_position_world)
        distance_difference = previous_distance_from_goal - current_distance_from_goal
        return np.exp2(distance_difference) if current_distance_from_goal < 0.1 * self.world_width else np.exp2(
            -current_distance_from_goal)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        tmp_state = self.state
        x, x_dot, theta, theta_dot = self.new_state(action)
        self.state = (x, x_dot, theta, theta_dot)
        self.previous_state = tmp_state

        done = not self.x_min <= x <= self.x_max or not self.theta_min <= theta <= self.theta_max  # TODO: test

        reward = self.reward()

        if done and not self.already_done:
            self.already_done = True
        elif done:
            if self.already_done:
                logger.warn('''You are calling 'step()' even though this environment has already returned done = True. 
                    You should always call 'reset()' once you receive 'done = True' 
                    -- any further steps are undefined behavior.''')
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):

        x, x_dot, theta, theta_dot = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # cart
            l, r, t, b = -self.cart_width / 2, self.cart_width / 2, self.cart_height / 2, -self.cart_height / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # track / ground
            xs = np.linspace(self.x_min, self.x_max, 2000)
            ys = self.height(xs)
            xys = list(zip((xs - self.x_min) * self.scale, ys * self.scale))

            self.track = rendering.make_polyline([(0, 0),
                                                  *xys,
                                                  (self.screen_width, 0)])
            self.track.set_linewidth(5)
            self.viewer.add_geom(self.track)

            # pole
            pole_line = LineString([(0, 0), (0, self.pole_length)]).buffer(self.pole_width / 2)
            pole = rendering.make_polygon(list(pole_line.exterior.coords))
            pole.set_color(0.8, 0.6, 0.4)
            self.pole_trans = rendering.Transform(translation=(0, self.cart_height / 2))
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

        self.cart_trans.set_translation((x - self.x_min) * self.scale, self.height(x) * self.scale)
        k = np.arctan(-1 / self.dheight(x))
        self.cart_trans.set_rotation(pi / 2 + k if k < 0 else k - pi / 2)

        self.pole_trans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
