from ctypes import *
import numpy as np
import os
import platform


def lib_path():
    dirname = os.path.dirname(__file__)

    system = platform.system()
    if system == 'Windows':
        lib_name = 'interf.dll'
    elif system == 'Darwin':
        lib_name = 'libinterf.dylib'
    else:
        lib_name = 'libinterf.so'

    return os.path.join(dirname, 'libs/' + lib_name)


libc = cdll.LoadLibrary(lib_path())

libc.calc_image.argtypes = [
    c_double, c_double, c_int,
    POINTER(c_double), POINTER(c_double), c_double,
    POINTER(c_double), POINTER(c_double), c_double,
    c_int, c_double, c_double, c_bool,
    c_int, POINTER(c_double)
]


def calc_image(
        start, end, n_points,
        wave_vector1, center1, radius1,
        wave_vector2, center2, radius2,
        n_frames, lamb, omega, has_interf,
        n_threads=8):

    print('calc_image_cpp')

    image = (c_double * (n_frames * n_points * n_points))()

    def to_double_pointer(nparray):
        return nparray.ctypes.data_as(POINTER(c_double))

    libc.calc_image(
        start, end, n_points,
        to_double_pointer(wave_vector1), to_double_pointer(center1), radius1,
        to_double_pointer(wave_vector2), to_double_pointer(center2), radius2,
        n_frames, lamb, omega, has_interf,
        n_threads, image
    )

    result = np.ctypeslib.as_array(image)
    result = result.reshape(n_frames, n_points, n_points)

    # to uint8
    im_min, im_max = 0, 4
    result = 255.0 * (result - im_min) / (im_max - im_min)
    return result.astype('uint8')
