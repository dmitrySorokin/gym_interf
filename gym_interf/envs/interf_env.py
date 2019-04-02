import gym
from matplotlib import pyplot as plt
from math import *
from cmath import *
import numpy as np
import time as tm

from .calc_image_cpp import calc_image as fast_calc_image
from .utils import reflect, project, rotate_euler_angles, dist


class InterfEnv(gym.Env):
    n_points = 64

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (0, 1)
    observation_space = gym.spaces.Box(low=0, high=4, shape=(1, n_points, n_points), dtype=np.float64)
    action_space = gym.spaces.Discrete(9)

    lamb = 8 * 1e-4
    omega = 1
    radius = 2.5

    # size of interferometer
    a = 100
    b = 200
    c = 100

    # image min & max coords
    x_min = -10
    x_max = 10

    # initial angles
    mirror1_angle = np.array([3*pi/4, 0, 0], dtype=np.float64)
    mirror2_angle = np.array([-pi/4, 0, 0], dtype=np.float64)

    min_distance = 1e-2
    max_distance = 10

    delta_angle = 1. / 1000

    reset_actions = 100

    def __init__(self):
        self.mirror1_angle = None
        self.mirror2_angle = None

        self.state = None
        self.info = {}

    def get_keys_to_action(self):
        return {
            (ord('w'),): 1,
            (ord('s'),): 2,
            (ord('a'),): 3,
            (ord('d'),): 4,
            (ord('i'),): 5,
            (ord('k'),): 6,
            (ord('j'),): 7,
            (ord('l'),): 8
        }

    def step(self, action):
        """

        :param action: (mirror_name, axis, delta_angle)
        :return: (state, reward, done, info)
        """
        center1, wave_vector1, center2, wave_vector2 = self._take_action(action)
        self.state = self._calc_state(center1, wave_vector1, center2, wave_vector2)

        distance = self._calc_projection_distance(center1, wave_vector1, center2, wave_vector2)
        reward = self._calc_reward()
        done = self._is_done(distance, reward)

        return self.state, reward, done, self.info

    def reset(self):
        self.mirror1_angle = np.copy(InterfEnv.mirror1_angle)
        self.mirror2_angle = np.copy(InterfEnv.mirror2_angle)

        for _ in range(InterfEnv.reset_actions):
            self._take_action(self.action_space.sample())
            c1, k1, c2, k2 = self._calc_centers_and_wave_vectors()
        self.state = self._calc_state(c1, k1, c2, k2)

        return self.state

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self.state
        elif mode == 'human':
            plt.imshow(self.state[0], vmin=0, vmax=4)
            plt.ion()
            plt.pause(1)
            plt.show()
        else:
            return None

    def _take_action(self, action):
        """
        0 - do nothing
        [1, 2, 3, 4] - mirror1
        [5, 6, 7, 8] - mirror2
        :param action:
        :return:
        """

        assert type(action) is int
        assert action in range(9)

        if action == 0:
            pass
        elif action == 1:
            self.mirror1_angle[0] += InterfEnv.delta_angle
        elif action == 2:
            self.mirror1_angle[0] -= InterfEnv.delta_angle
        elif action == 3:
            self.mirror1_angle[1] += InterfEnv.delta_angle
        elif action == 4:
            self.mirror1_angle[1] -= InterfEnv.delta_angle
        elif action == 5:
            self.mirror2_angle[0] += InterfEnv.delta_angle
        elif action == 6:
            self.mirror2_angle[0] -= InterfEnv.delta_angle
        elif action == 7:
            self.mirror2_angle[1] += InterfEnv.delta_angle
        elif action == 8:
            self.mirror2_angle[1] -= InterfEnv.delta_angle
        else:
            assert False

        return self._calc_centers_and_wave_vectors()

    def _calc_centers_and_wave_vectors(self):
        mirror1_normal = np.array([0, 0, 1])
        mirror1_normal = rotate_euler_angles(mirror1_normal, self.mirror1_angle)

        mirror2_normal = np.array([0, 0, 1])
        mirror2_normal = rotate_euler_angles(mirror2_normal, self.mirror2_angle)

        self.info['mirror1_normal'] = mirror1_normal
        self.info['mirror2_normal'] = mirror2_normal

        wave_vector1 = np.array([0, 0, 1], dtype=np.float64)
        center1 = np.array([0, 0, -InterfEnv.c], dtype=np.float64)

        center2 = np.array([0, -InterfEnv.a, -(InterfEnv.b + InterfEnv.c)], dtype=np.float64)
        wave_vector2 = np.array([0, 0, 1], dtype=np.float64)

        # reflect wave vector by first mirror
        center2 = project(center2, wave_vector2, mirror1_normal, np.array([0, -InterfEnv.a, -InterfEnv.c]))
        wave_vector2 = reflect(wave_vector2, mirror1_normal)
        self.info['reflect_with_mirror1'] = 'center = {}, k = {}'.format(center2, wave_vector2)

        # reflect wave vector by second mirror
        center2 = project(center2, wave_vector2, mirror2_normal, np.array([0, 0, -InterfEnv.c]))
        wave_vector2 = reflect(wave_vector2, mirror2_normal)
        self.info['reflect_with_mirror2'] = 'center = {}, k = {}'.format(center2, wave_vector2)

        return center1, wave_vector1, center2, wave_vector2

    def _calc_projection_distance(self, center1, wave_vector1, center2, wave_vector2):
        projection_plane_normal = np.array([0, 0, 1])
        projection_plane_center = np.array([0, 0, 0])

        proj_1 = project(center1, wave_vector1, projection_plane_normal, projection_plane_center)
        proj_2 = project(center2, wave_vector2, projection_plane_normal, projection_plane_center)
        distance = dist(proj_1, proj_2)

        self.info['proj_1'] = proj_1
        self.info['proj_2'] = proj_2
        self.info['dist'] = distance

        return distance

    def _calc_reward(self):
        max_pixels = self.state.max(axis=0)
        min_pixels = self.state.min(axis=0)
        visib = (max_pixels - min_pixels) / (max_pixels + min_pixels)

        #center = int(InterfEnv.n_points / 2)
        #radius_in_pixels = int(InterfEnv.radius * InterfEnv.n_points / (InterfEnv.x_max - InterfEnv.x_min))

        #min_pixels = center - radius_in_pixels // 2
        #max_pixels = center + radius_in_pixels // 2

        #print(min_pixels, max_pixels)

        #visib = visib[min_pixels: max_pixels, min_pixels: max_pixels]

        #print('max_pixels', max_pixels.shape)
        #print('min_pixels', min_pixels.shape)
        #print('intens', visib[30:34, 30:34])
        result = np.mean(visib)
        return result

    def _is_done(self, distance, visibility):
        return distance > InterfEnv.max_distance or visibility == 1

    def _calc_state(self, center1, wave_vector1, center2, wave_vector2):
        n_frames = 20
        state = np.ndarray(shape=(n_frames, InterfEnv.n_points, InterfEnv.n_points), dtype=np.float64)
        state_calc_time = 0

        for i, time in enumerate(np.linspace(0, 2 * pi, n_frames)):
            start = tm.time()

            image = fast_calc_image(
                InterfEnv.x_min, InterfEnv.x_max, InterfEnv.n_points,
                wave_vector1, center1, InterfEnv.radius,
                wave_vector2, center2, InterfEnv.radius,
                time, InterfEnv.lamb, InterfEnv.omega,
                n_threads=8)
            state[i] = image

            end = tm.time()

            state_calc_time += end - start

        self.info['state_calc_time'] = state_calc_time

        return state
