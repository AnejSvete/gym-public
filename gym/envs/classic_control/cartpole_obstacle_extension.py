import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
from scipy.constants import g, pi

from shapely.geometry import LineString, Polygon


def angle(ba, bc):
    return np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))


class CartPoleObstacleExtendedEnv(gym.Env):
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
        self.obstacle_width, self.obstacle_height = self.screen_width / 5, self.screen_height / 5  # TODO: convert to world coordinates
        self.obstacle_coordinates = [self.screen_width / 2 - self.obstacle_width / 2,
                                     self.screen_width / 2 + self.obstacle_width / 2,
                                     self.track_height + self.cart_height + self.pole_length + self.obstacle_height / 2 + 150,
                                     self.track_height + self.cart_height + self.pole_length - self.obstacle_height / 2 + 150]

        self.pole_length_world = self.pole_length / self.scale

        self.goal_position_world = 1.9

        self.intersection_polygon = None

        self.seed()
        self.viewer = None
        self.previous_state, self.state = None, None

        self.already_done = False

    def reset(self):
        self.state = self.np_random.uniform(low=(-2.0, -0.05, -0.05, -0.05),
                                            high=(-1.75, 0.05, 0.05, 0.05),
                                            size=(4,))
        self.already_done = False
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

    def pole_touches_obstacle(self):

        l, r, t, b = self.obstacle_coordinates
        obstacle = Polygon([(l, b), (l, t), (r, t), (r, b)])
        pole = LineString([self.pole_bottom_coordinates(), self.pole_top_coordinates()]).buffer(self.pole_width / 2)
        intersection = obstacle.intersection(pole)

        if intersection.is_empty:
            return False
        else:
            self.intersection_polygon = intersection
            return True

    def pole_obstacle_intersection(self):

        l, r, t, b = self.obstacle_coordinates
        obstacle = Polygon([(l, b), (l, t), (r, t), (r, b)])
        pole = LineString([self.pole_bottom_coordinates(), self.pole_top_coordinates()]).buffer(self.pole_width / 2)

        x, x_dot, theta, theta_dot = self.state
        # print(f'I am working with state {self.state}')
        pole_bottom = np.array([*self.pole_bottom_coordinates(screen_coordinates=True)])

        psi = pi / 2 - theta

        phi1 = angle(np.array([1, 0]), np.array([r, b]) - pole_bottom)
        phi2 = angle(np.array([1, 0]), np.array([l, b]) - pole_bottom)
        phi3 = angle(np.array([1, 0]), np.array([r, t]) - pole_bottom)
        phi4 = angle(np.array([1, 0]), np.array([l, t]) - pole_bottom)

        theta_old = theta

        cart_x = x * self.scale + self.screen_width / 2.0

        # eta = 0.01
        if cart_x < l:
            if psi < phi2:
                eta = 0.01
            else:
                eta = -0.01
        elif l <= cart_x <= r:
            if theta < 0.0:
                eta = -0.01
            else:
                eta = 0.01
        else:
            if psi < phi1:  # TODO: maybe not the right angles (into negative...)
                eta = 0.01
            else:
                eta = -0.01

        while not obstacle.intersection(pole).is_empty:
            theta += eta
            pole = LineString([self.pole_bottom_coordinates(),
                               self.pole_top_coordinates(provided_theta=theta)]).buffer(self.pole_width / 2)

        return theta

    def new_state(self, action):

        x, x_dot, theta, theta_dot = self.state
        # print('------------------------------')
        # print(f'the state is now {self.state}')

        if not self.pole_touches_obstacle():

            force = self.force_mag if action == 1 else -self.force_mag
            theta_dot_dot = self.theta_acc(force)

            x_dot_dot = self.x_acc(theta_dot_dot, force)

            x += self.tau * x_dot
            x_dot += self.tau * x_dot_dot
            theta += self.tau * theta_dot
            theta_dot += self.tau * theta_dot_dot
        else:

            force = self.force_mag if action == 1 else -self.force_mag
            theta_dot_dot = self.theta_acc(force)

            x_dot_dot = self.x_acc(theta_dot_dot, force)

            x += self.tau * x_dot
            x_dot += self.tau * x_dot_dot
            theta_dot += self.tau * theta_dot_dot

            self.state = (x, x_dot, theta, theta_dot)

            theta = self.pole_obstacle_intersection()

            x_dot_dot *= -1

            x += self.tau * x_dot
            x_dot += self.tau * x_dot_dot
            theta_dot += self.tau * theta_dot_dot

            # cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            # theta_dot_dot = ((self.gravity * sin_theta) /
            #                  (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)))
            # theta_dot_dot = -((self.gravity * sin_theta) /
            #                  (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)))
            theta_dot = 0

        # self.state = (x, x_dot, theta, theta_dot)
        # print(self.pole_touches_obstacle())

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
        return np.exp2(distance_difference) if current_distance_from_goal < 0.1 * self.world_width else np.exp2(-current_distance_from_goal)

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

        pole = LineString([self.pole_bottom_coordinates(), self.pole_top_coordinates()]).buffer(self.pole_width / 2)
        pole_polygon = rendering.FilledPolygon(list(pole.exterior.coords))
        pole_polygon.set_color(0.8, 0.6, 0.4)
        self.viewer.add_onetime(pole_polygon)

        if self.intersection_polygon is not None:
            intersection_polygon = rendering.FilledPolygon(list(self.intersection_polygon.exterior.coords))
            intersection_polygon.set_color(0.75, 0.75, 0.75)
            self.viewer.add_onetime(intersection_polygon)

        self.cart_trans.set_translation(x * self.scale + self.screen_width / 2, self.cart_y)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
