import gym
from matplotlib import pyplot as plt
from math import *
from cmath import *
import numpy as np
import time as tm
from scipy import optimize

from .calc_image_cpp import calc_image as calc_image_cpp
from .utils import reflect, project, rotate_x, rotate_y, dist, angle_between


class InterfEnv(gym.Env):
    n_points = 64
    n_frames = 16

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (0, 1)

    observation_space = gym.spaces.Box(low=0, high=4, shape=(n_frames, n_points, n_points), dtype=np.float64)
    action_space = gym.spaces.Discrete(8)

    lamb = 8 * 1e-4
    omega = 1
    radius = 1

    # size of interferometer
    a = 1000
    b = 200
    c = 10

    # image min & max coords
    x_min = -4
    x_max = 4

    # initial normals
    mirror1_x_rotation_angle = 3 * pi / 4
    mirror2_x_rotation_angle = -pi / 4

    # mirror screw step l / L, (ratio of delta screw length to vertical distance)
    mirror_screw_step = pi / 100000

    reset_actions = 50
    done_visibility = 0.9999

    max_steps = 4 * reset_actions

    def __init__(self):
        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        self.state = None
        self.n_steps = None
        self.info = None
        self.visib = None

        self._calc_reward = self._calc_reward_visib_minus_1
        self._calc_image = calc_image_cpp

    def set_calc_reward(self, method):
        if method == 'visib_minus_1':
            self._calc_reward = self._calc_reward_visib_minus_1
        elif method == 'delta_visib':
            self._calc_reward = self._calc_reward_delta_visib
        else:
            assert False, 'unknown reward_calc == {} optnions are "visib_minus1", "delta_visib"'.format(method)

    def set_calc_image(self, device):
        if device == 'cpu':
            self._calc_image = calc_image_cpp
        elif device == 'gpu':
            from .calc_image_cuda import calc_image as calc_image_gpu
            self._calc_image = calc_image_gpu
        else:
            assert False, 'unknown device == {} optnions are "cpu", "gpu"'.format(device)

    def get_keys_to_action(self):
        return {
            (ord('w'),): 0,
            (ord('s'),): 1,
            (ord('a'),): 2,
            (ord('d'),): 3,
            (ord('i'),): 4,
            (ord('k'),): 5,
            (ord('j'),): 6,
            (ord('l'),): 7
        }

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def step(self, action):
        """

        :param action: (mirror_name, axis, delta_angle)
        :return: (state, reward, done, info)
        """

        self.n_steps += 1

        self._take_action(action, InterfEnv.mirror_screw_step)
        center1, wave_vector1, center2, wave_vector2 = self._calc_centers_and_wave_vectors()
        self.state, tot_intens = self._calc_state(center1, wave_vector1, center2, wave_vector2)

        distance = self._calc_projection_distance(center1, wave_vector1, center2, wave_vector2)
        reward = self._calc_reward(tot_intens)

        return self.state, reward, self.game_over(), self.info

    def reset(self):
        self.n_steps = 0
        self.info = {}

        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        # fix me: use random step for every action
        for _ in range(InterfEnv.reset_actions):
            self._take_action(self.action_space.sample(), InterfEnv.mirror_screw_step)

        c1, k1, c2, k2 = self._calc_centers_and_wave_vectors()
        self.state, tot_intens = self._calc_state(c1, k1, c2, k2)

        # should be called after self._calc_state()
        self.visib = self._calc_visib(tot_intens)

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

    def _take_action(self, action, step_length):
        """
        0 - do nothing
        [1, 2, 3, 4] - mirror1
        [5, 6, 7, 8] - mirror2
        :param action:
        :return:
        """

        if action == 0:
            self.mirror1_screw_y -= step_length
        elif action == 1:
            self.mirror1_screw_y += step_length
        elif action == 2:
            self.mirror1_screw_x -= step_length
        elif action == 3:
            self.mirror1_screw_x += step_length
        elif action == 4:
            self.mirror2_screw_y -= step_length
        elif action == 5:
            self.mirror2_screw_y += step_length
        elif action == 6:
            self.mirror2_screw_x += step_length
        elif action == 7:
            self.mirror2_screw_x -= step_length
        else:
            assert False, 'unknown action = {}'.format(action)

    def _calc_centers_and_wave_vectors(self):
        mirror1_x_component = - self.mirror1_screw_x / np.sqrt(self.mirror1_screw_x ** 2 + 1)
        mirror1_y_component = - self.mirror1_screw_y / np.sqrt(self.mirror1_screw_y ** 2 + 1)
        mirror1_z_component = np.sqrt(1 - mirror1_x_component ** 2 - mirror1_y_component ** 2)
        mirror1_normal = np.array(
            [mirror1_x_component, mirror1_y_component, mirror1_z_component],
            dtype=np.float64
        )
        mirror1_normal = rotate_x(mirror1_normal, InterfEnv.mirror1_x_rotation_angle)

        mirror2_x_component = - self.mirror2_screw_x / np.sqrt(self.mirror2_screw_x ** 2 + 1)
        mirror2_y_component = - self.mirror2_screw_y / np.sqrt(self.mirror2_screw_y ** 2 + 1)
        mirror2_z_component = np.sqrt(1 - mirror2_x_component ** 2 - mirror2_y_component ** 2)
        mirror2_normal = np.array(
            [mirror2_x_component, mirror2_y_component, mirror2_z_component],
            dtype=np.float64
        )
        mirror2_normal = rotate_x(mirror2_normal, InterfEnv.mirror2_x_rotation_angle)

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

        self.info['angle_between_beams'] = angle_between(wave_vector1, wave_vector2)

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

    def _calc_reward_visib_minus_1(self, tot_intens):
        self.visib = self._calc_visib(tot_intens)
        self.info['visib'] = self.visib
        return self.visib - 1.

    def _calc_reward_delta_visib(self, tot_intens):
        prev_visib = self.visib
        self.visib = self._calc_visib(tot_intens)
        self.info['visib'] = self.visib
        return self.visib - prev_visib

    def _calc_visib(self, tot_intens):
        def visib(vmin, vmax):
            return (vmax - vmin) / (vmax + vmin)

        imin, imax = min(tot_intens), max(tot_intens)
        self.info['fit_time'] = 0
        self.info['imin'] = imin
        self.info['imax'] = imax

        return visib(float(min(tot_intens)), float(max(tot_intens)))

        def fit_func(x, a, b, phi):
            return a + b * np.cos(x + phi)

        try:
            tstart = tm.time()
            params, params_covariance = optimize.curve_fit(
                fit_func, np.linspace(0, 2 * pi, InterfEnv.n_frames),
                tot_intens,
                p0=[np.mean(tot_intens), np.max(tot_intens) - np.mean(tot_intens), 0])
            tend = tm.time()

            self.info['fit_time'] = tend - tstart

            a_param = params[0]
            b_param = abs(params[1])

            fmax = a_param + b_param
            fmin = max(a_param - b_param, 0)

            return visib(fmin, fmax)
        except RuntimeError:
            return visib(float(min(tot_intens)), float(max(tot_intens)))

    def game_over(self):
        return self.visib > InterfEnv.done_visibility or \
               self.n_steps >= InterfEnv.max_steps

    def _calc_state(self, center1, wave_vector1, center2, wave_vector2):
        state_calc_time = 0

        tstart = tm.time()

        band_width_x = InterfEnv.lamb / abs(wave_vector2[0])
        band_width_y = InterfEnv.lamb / abs(wave_vector2[1])
        band_width = min(band_width_x, band_width_y)
        cell_size = (InterfEnv.x_max - InterfEnv.x_min) / InterfEnv.n_points

        has_interf = band_width > 4 * cell_size

        #print('band_width_x = {}, band_width_y = {}, cell_size = {}, interf = {}'.format(
        #    band_width_x, band_width_y, cell_size, has_interf)
        #)

        state = self._calc_image(
            InterfEnv.x_min, InterfEnv.x_max, InterfEnv.n_points,
            wave_vector1, center1, InterfEnv.radius,
            wave_vector2, center2, InterfEnv.radius,
            InterfEnv.n_frames, InterfEnv.lamb, InterfEnv.omega,
            has_interf=has_interf)

        tend = tm.time()

        state_calc_time += tend - tstart

        self.info['state_calc_time'] = state_calc_time

        return state
