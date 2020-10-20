import gym
from matplotlib import pyplot as plt
from math import *
from cmath import *
import numpy as np
import time as tm
from scipy import optimize

from .calc_image_cpp import calc_image as calc_image_cpp
from .utils import reflect, project, rotate_x, rotate_y, dist, angle_between
from .domain_randomizer import DomainRandomizer
from .exp_state_provider import ExpStateProvider


class InterfEnv(gym.Env):
    n_points = 64
    n_frames = 16
    n_actions = 4

    # mirror screw step l / L, (ratio of delta screw length to vertical distance)
    one_step = 0.52 * 1e-6
    far_mirror_max_screw_value = one_step * 5000
    near_mirror_max_screw_value = one_step * 2500

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (0, 1)

    observation_space = gym.spaces.Box(low=0, high=255, shape=(n_frames, n_points, n_points), dtype=np.uint8)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float64)

    # in mm
    lamb = 6.35 * 1e-4
    omega = 1

    # size of interferometer (in mm)
    a = 200
    b = 300
    c = 100

    # initial normals
    mirror1_x_rotation_angle = 3 * pi / 4
    mirror2_x_rotation_angle = -pi / 4

    done_visibility = 0.9999


    def __init__(self):
        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        self.state = None
        self.n_steps = None
        self.info = None
        self.visib = None
        self.dist = None
        self.angle = None
        self.noise_coef = 0
        self.backward_frames = 4
        self.radius1 = 0.5
        self.radius2 = 1
        self.r_curvature = 5
        self.max_steps = 200

        self.beam1_mask = None
        self.beam2_mask = None

        self.beam1_rotation = 0
        self.beam2_rotation = 0
        self.beam1_sigmax = 1
        self.beam1_sigmay = 1
        self.beam2_sigmax = 1
        self.beam2_sigmay = 1

        self._calc_reward = self._calc_reward_visib_minus_1
        self._calc_image = calc_image_cpp
        self._image_randomizer = DomainRandomizer('data')
        self._use_beam_masks = False

        self._exp_state_provider = ExpStateProvider('saved_states')
        self._exp_state_provider.get_state()
        self._use_exp_data = False

        # image min & max coords
        self.x_min = -3.57 / 2
        self.x_max = 3.57 / 2

    def set_radius1(self, value):
        self.radius1 = value

    def set_radius2(self, value):
        self.radius2 = value

    def set_xmin(self, value):
        self.x_min = value

    def set_xmax(self, value):
        self.x_max = value

    def set_max_steps(self, value):
        self.max_steps = value

    def set_beam_rotation(self, value):
        self.beam1_rotation = value
        self.beam2_rotation = value

    def set_beam_ellipticity(self, value):
        self.beam1_sigmax = 1.0 / np.sqrt(value)
        self.beam1_sigmay = 1.0 * np.sqrt(value)
        self.beam2_sigmax = self.beam1_sigmax
        self.beam2_sigmay = self.beam1_sigmay


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

    def set_backward_frames(self, val):
        self.backward_frames = val

    def add_noise(self, noise_coef):
        self.noise_coef = noise_coef

    def use_beam_masks(self, enabled):
        self._use_beam_masks = enabled

    def use_exp_data(self, enabled):
        self._use_exp_data = enabled

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

    def step(self, actions):
        """

        :param action: (mirror_name, axis, delta_angle)
        :return: (state, reward, done, info)
        """

        self.n_steps += 1

        for action_id, action_value in enumerate(actions):
            self._take_action(action_id, action_value)

        center1, wave_vector1, center2, wave_vector2, self.angle = self._calc_centers_and_wave_vectors()
        self.state, tot_intens = self._calc_state(center1, wave_vector1, center2, wave_vector2)

        self.dist = self._calc_projection_distance(center1, wave_vector1, center2, wave_vector2)
        reward = self._calc_reward(tot_intens)

        return self.state, reward, self.game_over(), self.info

    def reset(self, actions=None):
        self.n_steps = 0
        self.info = {}
        self.beam1_mask = self._image_randomizer.get_mask()
        self.beam2_mask = self._image_randomizer.get_mask()

        self.mirror1_screw_x = 0
        self.mirror1_screw_y = 0
        self.mirror2_screw_x = 0
        self.mirror2_screw_y = 0

        if actions is None:
            actions = InterfEnv.action_space.sample()

        for action_id, action_value in enumerate(actions):
            self._take_action(action_id, action_value)

        c1, k1, c2, k2, self.angle = self._calc_centers_and_wave_vectors()
        self.state, tot_intens = self._calc_state(c1, k1, c2, k2)

        # should be called after self._calc_state()
        self.visib = self._calc_visib(tot_intens)
        self.dist = self._calc_projection_distance(c1, k1, c2, k2)

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

    def _take_action(self, action, normalized_step_length):
        """
        0 - do nothing
        [1, 2, 3, 4] - mirror1
        [5, 6, 7, 8] - mirror2
        :param action:
        :return:
        """

        if action == 0:
            self.mirror1_screw_x = np.clip(self.mirror1_screw_x + normalized_step_length, -1, 1)
        elif action == 1:
            self.mirror1_screw_y = np.clip(self.mirror1_screw_y + normalized_step_length, -1, 1)
        elif action == 2:
            self.mirror2_screw_x = np.clip(self.mirror2_screw_x + normalized_step_length, -1, 1)
        elif action == 3:
            self.mirror2_screw_y = np.clip(self.mirror2_screw_y + normalized_step_length, -1, 1)
        else:
            assert False, 'unknown action = {}'.format(action)

    def _calc_centers_and_wave_vectors(self):
        assert abs(self.mirror1_screw_x) <= 1, self.mirror1_screw_x
        assert abs(self.mirror1_screw_y) <= 1, self.mirror1_screw_y
        assert abs(self.mirror2_screw_x) <= 1, self.mirror2_screw_x
        assert abs(self.mirror2_screw_y) <= 1, self.mirror2_screw_y

        mirror1_screw_x_value = self.mirror1_screw_x * InterfEnv.far_mirror_max_screw_value
        mirror1_screw_y_value = self.mirror1_screw_y * InterfEnv.far_mirror_max_screw_value
        mirror1_x_component = - mirror1_screw_x_value / np.sqrt(mirror1_screw_x_value ** 2 + 1)
        mirror1_y_component = - mirror1_screw_y_value / np.sqrt(mirror1_screw_y_value ** 2 + 1)
        mirror1_z_component = np.sqrt(1 - mirror1_x_component ** 2 - mirror1_y_component ** 2)
        mirror1_normal = np.array(
            [mirror1_x_component, mirror1_y_component, mirror1_z_component],
            dtype=np.float64
        )
        mirror1_normal = rotate_x(mirror1_normal, InterfEnv.mirror1_x_rotation_angle)

        mirror2_screw_x_value = self.mirror2_screw_x * InterfEnv.near_mirror_max_screw_value
        mirror2_screw_y_value = self.mirror2_screw_y * InterfEnv.near_mirror_max_screw_value
        mirror2_x_component = - mirror2_screw_x_value / np.sqrt(mirror2_screw_x_value ** 2 + 1)
        mirror2_y_component = - mirror2_screw_y_value / np.sqrt(mirror2_screw_y_value ** 2 + 1)
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

        self.info['kvector'] = wave_vector2

        angle = angle_between(wave_vector1, wave_vector2)
        self.info['angle_between_beams'] = angle

        return center1, wave_vector1, center2, wave_vector2, angle

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
               self.n_steps >= self.max_steps

    def _calc_state(self, center1, wave_vector1, center2, wave_vector2):
        if self._use_exp_data:
            state, tot_intens, handles = self._exp_state_provider.get_state()
            self.mirror1_screw_x, self.mirror1_screw_y, self.mirror2_screw_x, self.mirror2_screw_y = handles
            print('-handles', -handles / 5000)
            self.info['state_calc_time'] = 0
            return state, tot_intens

        state_calc_time = 0

        tstart = tm.time()

        band_width_x = InterfEnv.lamb / abs(wave_vector2[0])
        band_width_y = InterfEnv.lamb / abs(wave_vector2[1])
        band_width = min(band_width_x, band_width_y)
        cell_size = (self.x_max - self.x_min) / InterfEnv.n_points

        has_interf = True#band_width > 4 * cell_size

        #print('band_width / (4 * cells_size)', band_width / (2 * cell_size))

        #print('band_width_x = {}, band_width_y = {}, cell_size = {}, interf = {}'.format(
        #    band_width_x, band_width_y, cell_size, has_interf)
        #)

        state = self._calc_image(
            self.x_min, self.x_max, InterfEnv.n_points,
            wave_vector1, center1, self.radius1, self.beam1_mask, 3.57, 64, self.beam1_sigmax, self.beam1_sigmay, 1.0, self.beam1_rotation,
            wave_vector2, center2, self.radius2, self.beam2_mask, 3.57, 64, self.beam2_sigmax, self.beam2_sigmay, 1.0, self.beam2_rotation, self.r_curvature,
            InterfEnv.n_frames - self.backward_frames, self.backward_frames, InterfEnv.lamb, InterfEnv.omega,
            noise_coef=self.noise_coef,
            use_beam_masks=self._use_beam_masks,
            has_interf=has_interf)

        tend = tm.time()

        state_calc_time += tend - tstart

        self.info['state_calc_time'] = state_calc_time

        return state
