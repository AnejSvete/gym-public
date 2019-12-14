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
        self.gravity = g
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = 1.0  # actually half the pole's length
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold = pi / 6
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

        self.cart_y = 100

        self.screen_width = 1200
        self.screen_height = 800

        self.cart_width = 100.0
        self.cart_height = 60.0
        self.track_height = self.cart_y - self.cart_height / 2

        self.world_width = self.x_threshold * 2
        self.scale = self.screen_width / self.world_width
        self.pole_width = 20.0
        self.pole_length = self.scale * (2 * self.length)
        self.obstacle_coordinates = [700, 1000,
                                     self.track_height + self.cart_height + self.pole_length + 100,
                                     self.track_height + self.cart_height + self.pole_length - 100]

        self.seed()
        self.viewer = None
        self.state, self.previous_state = None, None

        self.steps_beyond_done = -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def pole_obstacle_intersection(self):
        l, r, t, b = self.obstacle_coordinates

        x0, y0 = self.pole_top_coordinates(previous=True)
        x1, y1 = self.pole_top_coordinates(previous=False)

        f = lambda z: (y1 - y0) / (x1 - x0) * (z - x0) + y0
        g = lambda z: (z - y0) * (x1 - x0) / (y1 - y0) + x0

        if x0 < l <= x1 and b <= y1 <= t:
            x, y = l, f(l)
        elif x0 > r >= x1 and b <= y1 <= t:
            x, y = r, f(r)
        elif l <= x0 <= r and l <= x1 <= r and y0 < b <= y1:
            x, y = g(b), b
        else:
            return None

        return x, y

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

    def pole_top_coordinates(self, previous=False):  # TODO: should be calculated from line?
        x, x_dot, theta, theta_dot = self.state if not previous or self.previous_state is None else self.previous_state
        return (x * self.scale + self.screen_width / 2.0 + self.pole_length * np.sin(theta),
                self.track_height + self.cart_height + self.pole_length * np.cos(theta))

    def pole_bottom_coordinates(self, previous=False):
        x, x_dot, theta, theta_dot = self.state if not previous or self.previous_state is None else self.previous_state
        return x * self.scale + self.screen_width / 2.0, self.track_height + self.cart_height

    def new_state(self, action):

        x, x_dot, theta, theta_dot = self.state

        intersection = self.pole_obstacle_intersection()
        if intersection is None:

            force = self.force_mag if action == 1 else -self.force_mag
            theta_dot_dot = self.theta_acc(force)

        else:

            force = 0
            intersection_x, intersection_y = intersection

            cart_x = x * self.scale + self.screen_width / 2.0
            axle_x, axle_y = cart_x, self.track_height + self.cart_height
            top_x, top_y = cart_x, intersection_y

            a, b = intersection_x - top_x, top_y - axle_y
            c = np.sqrt(a ** 2 + b ** 2)

            cos_theta = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
            tmp_theta = np.arccos(cos_theta)

            theta = tmp_theta if cart_x < intersection_x else -tmp_theta

            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            theta_dot_dot = -((self.gravity * sin_theta) /
                              (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)))
            theta_dot = 0

        x_dot_dot = self.x_acc(theta_dot_dot, force)

        x += self.tau * x_dot
        x_dot += self.tau * x_dot_dot
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_dot_dot

        return x, x_dot, theta, theta_dot

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        tmp_state = self.state
        x, x_dot, theta, theta_dot = self.new_state(action)
        self.state = (x, x_dot, theta, theta_dot)
        self.previous_state = tuple(tmp_state)

        done = x < -self.x_threshold or x > self.x_threshold or not -self.theta_threshold <= theta <= self.theta_threshold

        if not done:
            reward = 1.0
        elif self.steps_beyond_done == -1:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn('''You are calling 'step()' even though this environment has already returned done = True. 
                    You should always call 'reset()' once you receive 'done = True' 
                    -- any further steps are undefined behavior.''')
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = -1
        return np.array(self.state)

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

            # obstacle
            l, r, t, b = self.obstacle_coordinates
            obstacle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            obstacle.set_color(255, 0, 0)
            self.viewer.add_geom(obstacle)

            # pole head
            self.pole_head = rendering.make_circle(self.pole_width / 2)
            self.pole_head.set_color(0, 0, 255)
            pole_top = self.pole_top_coordinates()
            self.pole_top_trans = rendering.Transform(translation=(pole_top[0], pole_top[1]))
            self.pole_head.add_attr(self.pole_top_trans)
            self.viewer.add_geom(self.pole_head)

            ll = rendering.Line((0.0, self.cart_y), (self.screen_width, self.cart_y))
            self.viewer.add_geom(ll)

        if self.state is None:
            return None

        pole = LineString([self.pole_bottom_coordinates(), self.pole_top_coordinates()]).buffer(self.pole_width / 2)
        pole_polygon = rendering.FilledPolygon(list(pole.exterior.coords))
        pole_polygon.set_color(0.8, 0.6, 0.4)
        self.viewer.add_onetime(pole_polygon)

        cart_x = x * self.scale + self.screen_width / 2.0
        self.carttrans.set_translation(cart_x, self.cart_y)

        pole_top = self.pole_top_coordinates()
        self.pole_top_trans.set_translation(*pole_top)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
