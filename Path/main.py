import matplotlib.pyplot as plt
import numpy as np

from Path.track_uniform import *
from Path.Path_concat import *
from Path.draw_function import *

amplitude = np.arange(-0.5, 0.5, 0.015)
frequency = np.arange(0, 3.14, 0.06)
time_shift = np.arange(3.14, 3.14*2, 0.1)


def sinusoid(ampl, freq, shift):
    x = np.arange(0, 5, 5 / 750)

    x_amp = np.random.choice(ampl, 30)
    x_freq = np.random.choice(freq, 30)
    x_time = np.random.choice(shift, 30)

    func = 0

    for i in range(30):
        func += x_amp[i] * np.sin(x_freq[i] * x + x_time[i])

    return func


def func_transform(func, ampl, initial_pos):
    func *= ampl / max(func)
    func = func - min(func)
    func /= 1.5
    func += (initial_pos - func[0])
    return func


def make_path():
    idx = 0
    while True:
        x_function = sinusoid(amplitude, frequency, time_shift)
        y_function = sinusoid(amplitude, frequency, time_shift)

        x_function = func_transform(x_function, 800, 960)
        y_function = func_transform(y_function, 600, 540)

        if idx == 20:
            break
        if min(x_function) < 0 or min(y_function) < 0 or max(x_function) > 1850 or max(y_function) > 950:
            continue
        else:
            path = np.concatenate([x_function.reshape(-1, 1), y_function.reshape(-1, 1)], axis=1)

        _, dt_r = draw_dist(path, plot='False')

        if max(dt_r) < 4:
            print(max(dt_r))
            idx += 1
            print('=' * 50)
            print('Path : {}, Path Saved !!!!'.format(idx))
            print('=' * 50)
            draw_dist(path, idx, img='True')
            np.save('./Path_{}.npy'.format(idx), path)

    concat_path()

    draw_dist('./total_path.npy', '(total)', img='True')

    draw_track()

    track_direction()


if __name__ == '__main__':
    make_path()