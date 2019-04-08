import gym
from matplotlib import pyplot as plt
from math import *
from cmath import *
import numpy as np
import time as tm
from scipy import optimize

from .calc_image_cpp import calc_image as fast_calc_image
from .utils import reflect, project, rotate_x, rotate_y, dist


class InterfEnv(gym.Env):
    n_points = 256
    n_frames = 16

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (0, 1)
    observation_space = gym.spaces.Box(low=0, high=4, shape=(n_frames, n_points, n_points), dtype=np.float64)
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

    # initial normals
    mirror1_normal = rotate_x(np.array([0, 0, 1]), 3 * pi / 4)
    mirror2_normal = rotate_x(np.array([0, 0, 1]), -pi / 4)

    min_distance = 1e-2
    max_distance = 10

    delta_angle = pi / 100000

    reset_actions = 50
    done_visibility = 1

    max_steps = 200 * reset_actions

    def __init__(self):
        self.mirror1_normal = None
        self.mirror2_normal = None

        self.state = None
        self.n_steps = None
        self.info = None
        self.visib = None

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

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def step(self, action):
        """

        :param action: (mirror_name, axis, delta_angle)
        :return: (state, reward, done, info)
        """
        center1, wave_vector1, center2, wave_vector2 = self._take_action(action)
        self.state = self._calc_state(center1, wave_vector1, center2, wave_vector2)

        #distance = self._calc_projection_distance(center1, wave_vector1, center2, wave_vector2)
        reward = self._calc_reward()
        done = self._is_done()

        self.n_steps += 1

        return self.state, reward, done, self.info

    def reset(self):
        self.mirror1_normal = np.copy(InterfEnv.mirror1_normal)
        self.mirror2_normal = np.copy(InterfEnv.mirror2_normal)

        self.n_steps = 0
        self.info = {}

        for _ in range(InterfEnv.reset_actions):
            self._take_action(self.action_space.sample())

        c1, k1, c2, k2 = self._calc_centers_and_wave_vectors()
        self.state = self._calc_state(c1, k1, c2, k2)

        # should be called after self._calc_state()
        self.visib = self._calc_visib()

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

        assert action in range(9)

        if action == 0:
            pass
        elif action == 1:
            self.mirror1_normal = rotate_x(self.mirror1_normal, InterfEnv.delta_angle)
        elif action == 2:
            self.mirror1_normal = rotate_x(self.mirror1_normal, -InterfEnv.delta_angle)
        elif action == 3:
            self.mirror1_normal = rotate_y(self.mirror1_normal, InterfEnv.delta_angle)
        elif action == 4:
            self.mirror1_normal = rotate_y(self.mirror1_normal, -InterfEnv.delta_angle)
        elif action == 5:
            self.mirror2_normal = rotate_x(self.mirror2_normal, InterfEnv.delta_angle)
        elif action == 6:
            self.mirror2_normal = rotate_x(self.mirror2_normal, -InterfEnv.delta_angle)
        elif action == 7:
            self.mirror2_normal = rotate_y(self.mirror2_normal, InterfEnv.delta_angle)
        elif action == 8:
            self.mirror2_normal = rotate_y(self.mirror2_normal, -InterfEnv.delta_angle)
        else:
            assert False



        return self._calc_centers_and_wave_vectors()

    def _calc_centers_and_wave_vectors(self):
        self.info['mirror1_normal'] = self.mirror1_normal
        self.info['mirror2_normal'] = self.mirror2_normal

        wave_vector1 = np.array([0, 0, 1], dtype=np.float64)
        center1 = np.array([0, 0, -InterfEnv.c], dtype=np.float64)

        center2 = np.array([0, -InterfEnv.a, -(InterfEnv.b + InterfEnv.c)], dtype=np.float64)
        wave_vector2 = np.array([0, 0, 1], dtype=np.float64)

        # reflect wave vector by first mirror
        center2 = project(center2, wave_vector2, self.mirror1_normal, np.array([0, -InterfEnv.a, -InterfEnv.c]))
        wave_vector2 = reflect(wave_vector2, self.mirror1_normal)
        self.info['reflect_with_mirror1'] = 'center = {}, k = {}'.format(center2, wave_vector2)

        # reflect wave vector by second mirror
        center2 = project(center2, wave_vector2, self.mirror2_normal, np.array([0, 0, -InterfEnv.c]))
        wave_vector2 = reflect(wave_vector2, self.mirror2_normal)
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
        prev_visib = self.visib
        self.visib = self._calc_visib()
        self.info['visib'] = self.visib
        return self.visib - prev_visib

    def _calc_visib(self):
        tot_intens = [np.sum(image) for image in self.state]

        def fit_func(x, a, b, phi):
            return a + b * np.cos(x + phi)

        tstart = tm.time()
        params, params_covariance = optimize.curve_fit(
            fit_func, np.linspace(0, 2 * pi, InterfEnv.n_frames),
            tot_intens,
            p0=[np.mean(tot_intens), np.std(tot_intens), 0])
        tend = tm.time()

        self.info['fit_time'] = tend - tstart

        fmax = params[0] + fabs(params[1])
        fmin = max(params[0] - fabs(params[1]), 0)

        def visib(vmin, vmax):
            return (vmax - vmin) / (vmax + vmin)

        #return (imax - imin) / (imax + imin)
        return visib(fmin, fmax)

    def _is_done(self):
        return self.visib > InterfEnv.done_visibility or \
               self.n_steps > InterfEnv.max_steps

    def _calc_state(self, center1, wave_vector1, center2, wave_vector2):
        state_calc_time = 0

        tstart = tm.time()

        band_width_x = InterfEnv.lamb / abs(wave_vector2[0])
        band_width_y = InterfEnv.lamb / abs(wave_vector2[1])
        band_width = min(band_width_x, band_width_y)
        cell_size = (InterfEnv.x_max - InterfEnv.x_min) / InterfEnv.n_points

        has_interf = band_width > 4 * cell_size

        print('band_width_x = {}, band_width_y = {}, cell_size = {}, interf = {}'.format(
            band_width_x, band_width_y, cell_size, has_interf)
        )

        state = fast_calc_image(
            InterfEnv.x_min, InterfEnv.x_max, InterfEnv.n_points,
            wave_vector1, center1, InterfEnv.radius,
            wave_vector2, center2, InterfEnv.radius,
            InterfEnv.n_frames, InterfEnv.lamb, InterfEnv.omega,
            has_interf=has_interf, n_threads=8)

        tend = tm.time()

        state_calc_time += tend - tstart

        self.info['state_calc_time'] = state_calc_time

        return state
