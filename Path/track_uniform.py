import matplotlib.pyplot as plt
import numpy as np


def draw_dist(file, idx=False, plot=False, img=False):
    if '/' in file:
        tmp = np.load(file)
    else:
        tmp = file

    prev_x = tmp[:, 0]
    prev_y = tmp[:, 1]

    curr_x = np.zeros_like(prev_x)
    curr_y = np.zeros_like(prev_y)

    for i in range(0, len(prev_x)-1):
        curr_x[i+1] = prev_x[i]
        curr_y[i+1] = prev_y[i]

    dt_x = prev_x - curr_x
    dt_y = prev_y - curr_y

    dt_x = dt_x.reshape(-1, 1)
    dt_y = dt_y.reshape(-1, 1)

    dt = np.concatenate([dt_x, dt_y], axis=1)[1:]

    dt_tan = []
    for i in range(len(dt)):
        if dt[i][0] >= 0:
            dt_tan.append(np.rad2deg(np.arctan(dt[i][1]/dt[i][0])))
        elif dt[i][0] < 0:
            dt_tan.append(180 + np.rad2deg(np.arctan(dt[i][1]/dt[i][0])))
    dt_degree = np.array(dt_tan).astype('int')
    dt_degree += 90

    dt_r = []
    for i in range(len(dt)):
        dt_r.append(np.sqrt(dt[i][0] ** 2 + dt[i][1] ** 2))

    if plot == 'False':
        a = plt.hist(dt_degree, bins=360)

        return a, dt_r

    elif plot == 'True':
        plt.figure(figsize=(22, 8))
        plt.hist(dt_degree, bins=36)
        plt.xticks(list(range(0, 361, 10)))
        plt.show()

    if img == 'True':
        plt.figure(figsize=(22, 8))
        plt.hist(dt_degree, bins=36)
        plt.xticks(list(range(0, 361, 10)))
        plt.savefig('./Path{}_distribution.jpg'.format(idx))
        plt.close()

    else:
        pass

    return dt_r


if __name__ == '__main__':
    draw_dist('./total_path.npy', idx='total', plot='True', img='True')