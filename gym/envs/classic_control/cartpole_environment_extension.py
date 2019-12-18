import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi

from shapely.geometry import LineString


class CartPoleExtendedEnv(gym.Env):
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

        # self.theta_min, self.theta_max = -pi / 6, pi / 6
        self.theta_min, self.theta_max = -12 * 2 * pi / 360, 12 * 2 * pi / 360
        self.x_min, self.x_max = -self.world_width / 2, self.world_width / 2

        low = np.array([self.x_min * 2, -np.finfo(np.float32).max, self.theta_min * 2, -np.finfo(np.float32).max])
        high = np.array([self.x_max * 2, np.finfo(np.float32).max, self.theta_max * 2, np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
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

        self.pole_length_world = self.pole_length / self.scale

        self.goal_position_world = 1.6

        self.seed()
        self.viewer = None
        self.state, self.previous_state = None, None

        self.time = 0
        self.already_done = False

    def reset(self):
        self.state = self.np_random.uniform(low=(-2.0 + 1.0, -0.05, -0.05, -0.05),
                                            high=(-1.8 + 1.0, 0.05, 0.05, 0.05),
                                            size=(4,))
        self.already_done = False
        self.time = 0
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def theta_acc(self, force):
        x, x_dot, theta, theta_dot = self.state
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        theta_dot_dot = ((self.gravity * sin_theta - cos_theta *
                          (force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass) /
                         (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)))
        return theta_dot_dot

    def x_acc(self, theta_dot_dot, force):
        x, x_dot, theta, theta_dot = self.state
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_dot_dot = ((force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass -
                     self.pole_mass_length * theta_dot_dot * cos_theta / self.total_mass)
        return x_dot_dot

    def pole_top_coordinates(self, previous=False, screen_coordinates=True):
        x, x_dot, theta, theta_dot = self.state if not previous or self.previous_state is None else self.previous_state
        if screen_coordinates:
            return (x * self.scale + self.screen_width / 2 + self.pole_length * np.sin(theta),
                    self.track_height + self.cart_height + self.pole_length * np.cos(theta))
        else:
            return (x + self.pole_length_world * np.sin(theta),
                    self.track_height_world + self.cart_height_world + self.pole_length_world * np.cos(theta))

    def pole_bottom_coordinates(self, previous=False, screen_coordinates=True):
        x, x_dot, theta, theta_dot = self.state if not previous or self.previous_state is None else self.previous_state
        if screen_coordinates:
            return x * self.scale + self.screen_width / 2, self.track_height + self.cart_height
        else:
            return x, self.track_height_world + self.cart_height_world

    def new_state(self, action):

        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag
        theta_dot_dot = self.theta_acc(force)

        x_dot_dot = self.x_acc(theta_dot_dot, force)

        x += self.tau * x_dot
        x_dot += self.tau * x_dot_dot
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_dot_dot

        return np.array([x, x_dot, theta, theta_dot])

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.time += 1

        tmp_state = self.state
        x, x_dot, theta, theta_dot = self.new_state(action)
        self.state = (x, x_dot, theta, theta_dot)
        self.previous_state = tmp_state

        done = not self.x_min <= x <= self.x_max or not self.theta_min <= theta <= self.theta_max  # TODO: test

        distance_from_goal = np.abs(x - self.goal_position_world) + 1

        if self.time > 200:
            print(f'time... {self.time}')

        if not done:
            # reward = 1.0
            reward = 1.0 / np.log(distance_from_goal) if self.time > 200 else 1.0
        elif not self.already_done:
            self.already_done = True
            # reward = 1.0
            reward = 1.0 / np.log(distance_from_goal) if self.time > 200 else 1.0
        else:
            if self.already_done:
                logger.warn('''You are calling 'step()' even though this environment has already returned done = True. 
                    You should always call 'reset()' once you receive 'done = True' 
                    -- any further steps are undefined behavior.''')
            reward = 0.0

        # print(reward)

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):

        x, x_dot, theta, theta_dot = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # cart
            l, r, t, b = -self.cart_width / 2, self.cart_width / 2, self.cart_height / 2, -self.cart_height / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # axle
            self.axle = rendering.make_circle(self.pole_width / 2)
            self.axletrans = rendering.Transform(translation=(0, self.cart_height / 2))
            self.axle.add_attr(self.axletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            # track / ground
            self.track = rendering.FilledPolygon([(0, 0),
                                                  (0, self.track_height),
                                                  (self.screen_width, self.track_height),
                                                  (self.screen_width, 0)])
            self.track.set_color(0, 255, 0)
            self.viewer.add_geom(self.track)

            # pole head
            self.pole_head = rendering.make_circle(self.pole_width / 2)
            self.pole_head.set_color(0, 0, 255)
            pole_top = self.pole_top_coordinates()
            self.pole_top_trans = rendering.Transform(translation=(pole_top[0], pole_top[1]))
            self.pole_head.add_attr(self.pole_top_trans)
            self.viewer.add_geom(self.pole_head)

        if self.state is None:
            return None

        pole = LineString([self.pole_bottom_coordinates(), self.pole_top_coordinates()]).buffer(self.pole_width / 2)
        pole_polygon = rendering.FilledPolygon(list(pole.exterior.coords))
        pole_polygon.set_color(0.8, 0.6, 0.4)
        self.viewer.add_onetime(pole_polygon)

        self.carttrans.set_translation(x * self.scale + self.screen_width / 2, self.cart_y)

        pole_top = self.pole_top_coordinates()
        self.pole_top_trans.set_translation(*pole_top)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
