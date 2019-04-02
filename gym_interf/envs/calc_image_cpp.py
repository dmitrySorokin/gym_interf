from ctypes import *
import numpy as np
import os
import platform


dirname = os.path.dirname(__file__)

system = platform.system()
if system == 'Windows':
	lib_name = 'interf.dll'
else:
	lib_name = 'libinterf.so'

lib_path = os.path.join(dirname, 'libs/' + lib_name)

libc = cdll.LoadLibrary(lib_path)

libc.calc_image.argtypes = [
    c_double, c_double, c_int,
    c_void_p, c_void_p, c_double,
    c_void_p, c_void_p, c_double,
    c_double, c_double, c_double,
    c_int, c_void_p
]


def calc_image(
        start, end, n_points,
        wave_vector1, center1, radius1,
        wave_vector2, center2, radius2,
        time, lamb, omega,
        n_threads):

    image = (c_double * n_points * n_points)()

    libc.calc_image(
        start, end, n_points,
        c_void_p(wave_vector1.ctypes.data), c_void_p(center1.ctypes.data), radius1,
        c_void_p(wave_vector2.ctypes.data), c_void_p(center2.ctypes.data), radius2,
        time, lamb, omega,
        n_threads, image
    )

    return np.ctypeslib.as_array(image, shape=(n_points, n_points)).reshape(1, n_points, n_points)
