import numpy as np

cur_pos = []
tar_pos = []

for i in range(100000):
    cur_x = np.random.randint(0, 1850, 1)
    cur_y = np.random.randint(0, 950, 1)
    cur_pos.append(np.array([cur_x, cur_y]))

cur_pos = np.array(cur_pos)

for i in range(100000):
    tar_x = np.random.randint(0, 1850, 1)
    tar_y = np.random.randint(0, 950, 1)
    tar_pos.append(np.array([tar_x, tar_y]))

tar_pos = np.array(tar_pos)

np.save('../Data/cur_pos.npy', cur_pos)
np.save('../Data/target_pos.npy', tar_pos)