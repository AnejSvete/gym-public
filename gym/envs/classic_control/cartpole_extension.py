import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.constants import g, pi
from scipy.spatial.distance import cdist
from shapely.geometry import *


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
        self.theta_threshold_radians = pi / 6
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

        self.screen_width = 1200
        self.screen_height = 800

        self.cart_width = 100.0
        self.cart_height = 60.0
        self.axle_offset = self.cart_height / 4.0

        self.world_width = self.x_threshold * 2
        self.scale = self.screen_width / self.world_width
        self.cart_y = 100  # TOP OF CART
        self.pole_width = 20.0
        self.pole_length = self.scale * (2 * self.length)
        self.obstacle_coordinates = [700, 1000,
                                     self.cart_y + self.axle_offset + self.pole_length + 100,
                                     self.cart_y + self.axle_offset + self.pole_length - 100]

        self.seed()
        self.viewer = None
        self.state = None

        # self.intersection_polygon = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def pole_obstacle_intersection(self):
        l, r, t, b = self.obstacle_coordinates
        obstacle = Polygon([(l, b), (l, t), (r, t), (r, b)])
        pole = LineString([self.pole_bottom_coordinates(), self.pole_top_coordinates()]).buffer(self.pole_width / 2)
        print(obstacle)
        print(pole)
        intersection = obstacle.intersection(pole)
        if not intersection.is_empty:
            self.intersection_polygon = intersection.exterior.coords
        return intersection.centroid

    def extract_point(self, x, theta, intersection_coordinates):
        pole_top = self.pole_top_coordinates()
        l, r, t, b = self.obstacle_coordinates
        points = [(l, b), (l, t), (r, t), (r, b), pole_top]
        a = np.ndarray((1, 2), buffer=np.array(intersection_coordinates))
        b = np.ndarray((len(points), 2), buffer=np.array(points))
        ix = cdist(a, b).argmin()
        options = ['lb', 'lb', 'rt', 'rb', 'pole_top']
        print(f'chosen {options[ix]}')
        return points[ix]

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

    def pole_top_coordinates(self):
        x, x_dot, theta, theta_dot = self.state
        return (x * self.scale + self.screen_width / 2.0 + self.pole_length * np.sin(theta),
                self.cart_y + self.axle_offset + self.pole_length * np.cos(theta))

    def pole_bottom_coordinates(self):
        x, x_dot, theta, theta_dot = self.state
        return x * self.scale + self.screen_width / 2.0, self.cart_y + self.axle_offset

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        x, x_dot, theta, theta_dot = self.state

        intersection = self.pole_obstacle_intersection()
        if intersection.is_empty:

            force = self.force_mag if action == 1 else -self.force_mag
            theta_dot_dot = self.theta_acc(force)

        else:

            force = 0

            intersection_coordinates = self.extract_point(x, theta, (intersection.xy[0][0], intersection.xy[1][0]))
            print(f'intersection at {intersection_coordinates}')

            cart_x = x * self.scale + self.screen_width / 2.0
            axle_coordinates = (cart_x, self.cart_y + self.axle_offset)
            top_coordinates = (cart_x, intersection_coordinates[1])

            a = intersection_coordinates[0] - top_coordinates[0]
            b = top_coordinates[1] - axle_coordinates[1]
            c = np.sqrt(a ** 2 + b ** 2)

            cos_theta = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
            tmp_theta = np.arccos(cos_theta)

            if cart_x < intersection_coordinates[0]:
                theta = tmp_theta
            else:
                theta = -tmp_theta

            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            theta_dot_dot = ((self.gravity * sin_theta) /
                             (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)))
            theta_dot = 0

        x_dot_dot = self.x_acc(theta_dot_dot, force)

        x += self.tau * x_dot
        x_dot += self.tau * x_dot_dot
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_dot_dot

        self.state = (x, x_dot, theta, theta_dot)
        # self.state = (x + 0.005, 0, theta, 0)

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):

        x, x_dot, theta, theta_dot = self.state

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # cart
            l, r, t, b = -self.cart_width / 2, self.cart_width / 2, self.cart_height / 2, -self.cart_height / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # pole
            l, r, t, b = -self.pole_width / 2, self.pole_width / 2, self.pole_length - self.pole_width / 2, -self.pole_width / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, self.axle_offset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            # axle
            self.axle = rendering.make_circle(self.pole_width / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            # track
            self.track = rendering.Line((0, self.cart_y), (self.screen_width, self.cart_y))
            self.track.set_color(0, 0, 0)
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

            # intersect = rendering.FilledPolygon([(0, 0), (0, 0), (0, 0), (0, 0)])
            # intersect.set_color(0, 0, 255)
            # self.viewer.add_geom(intersect)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -self.pole_width / 2, self.pole_width / 2, self.pole_length - self.pole_width / 2, -self.pole_width / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cart_x = x * self.scale + self.screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cart_x, self.cart_y)
        self.poletrans.set_rotation(-theta)

        pole_top = self.pole_top_coordinates()
        self.pole_top_trans.set_translation(*pole_top)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
