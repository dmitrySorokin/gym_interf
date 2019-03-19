import gym
from matplotlib import pyplot as plt
from math import *
from cmath import *
import numpy as np
import time as tm

from .calc_image_cpp import calc_image as fast_calc_image
from .utils import reflect, project, rotate_euler_angles


class InterfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    lamb = 8 * 1e-4
    omega = 1
    radius1 = 2.5
    radius2 = 2.5

    # size of interferometer
    a = 100
    b = 200
    c = 100

    def __init__(self):
        self.mirror1_angle = np.array([135, 0, 0], dtype=np.float64)
        self.mirror2_angle = np.array([-45, 0, 0], dtype=np.float64)

        self.time = 1
        self.image = None

    def step(self, action):
        self.mirror1_angle[0] += 0.1

        mirror1_normal = np.array([0, 0, 1])
        mirror1_normal = rotate_euler_angles(mirror1_normal, self.mirror1_angle / 180 * pi)

        mirror2_normal = np.array([0, 0, 1])
        mirror2_normal = rotate_euler_angles(mirror2_normal, self.mirror2_angle / 180 * pi)

        print(mirror1_normal, mirror2_normal)

        wave_vector1 = np.array([0, 0, 1], dtype=np.float64)
        center1 = np.array([0, 0, -InterfEnv.c], dtype=np.float64)

        center2 = np.array([0, -InterfEnv.a, -(InterfEnv.b + InterfEnv.c)], dtype=np.float64)
        wave_vector2 = np.array([0, 0, 1], dtype=np.float64)

        # reflect wave vector by first mirror
        center2 = project(center2, wave_vector2, mirror1_normal, np.array([0, -InterfEnv.a, -InterfEnv.c]))
        wave_vector2 = reflect(wave_vector2, mirror1_normal)
        print('reflect_with mirror1: center = {}, k = {}'.format(center2, wave_vector2))

        # reflect wave vector by second mirror
        center2 = project(center2, wave_vector2, mirror2_normal, np.array([0, 0, -InterfEnv.c]))
        wave_vector2 = reflect(wave_vector2, mirror2_normal)
        print('reflect_with mirror2: center = {}, k = {}'.format(center2, wave_vector2))

        start = tm.time()
        self.image = fast_calc_image(
            -5, 5, 200,
            wave_vector1, center1, InterfEnv.radius1,
            wave_vector2, center2, InterfEnv.radius2,
            self.time, InterfEnv.lamb, InterfEnv.omega,
            8)
        end = tm.time()
        print('image calculation time {} s'.format(end - start))

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        plt.imshow(self.image, vmin=0, vmax=4)
        plt.ion()
        plt.pause(0.001)
        plt.show()
