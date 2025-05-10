import numpy as np


def concat_path():

    tmp_1 = np.load('./Path_1.npy')
    tmp_1 = tmp_1
    for idx in range(2, 21, 1):
        tmp_2 = np.load('Path_{}.npy'.format(idx))
        tmp_2 = tmp_2
        tmp_1 = np.concatenate([tmp_1, tmp_2], axis=0)
    print(tmp_1.shape)
    np.save('./total_path.npy', tmp_1)
    print('=' * 50)
    print('Path concatenated...')
    print('=' * 50)


if __name__ == '__main__':
    concat_path()