import time

import pygame.image

from DDPG.game_env.function.cursor import *
from DDPG.game_env.function.screen import *
from DDPG.game_env.function.target import *
from DDPG.game_env.function.event import *
from DDPG.game_env.function.parameters import *


def track_direction():
    _param = Parameters()  # Call the variables
    path = np.load('./total_path.npy')
    _screen = Screen()
    _target = Target()
    _cursor = Cursor()
    _event = Event()
    cur_list = []
    color_list = [(255, 0, 0), (125, 10, 10), (10, 255, 10), (10, 10, 255), (10, 125, 10), (10, 10, 125),
                      (125, 125, 10),
                      (125, 10, 125), (10, 125, 125), (125, 125, 125)]

    _screen.overwrite()  # Fill the background with black color.
    for i in range(path.shape[0]):
        cur_list.append([path[i][0], path[i][1]])
    a = 0

    for idx, (cur_x, cur_y) in enumerate(cur_list):
        if idx % 1500 < 10:
            G.draw.circle(_screen.screen, G.Color(color_list[a]), (cur_x, cur_y), 1.5, 3)
        if idx != 0 and idx % 1500 == 0:
            a += 1
        if a >= 10:
            a -= 10

    flip()  # Flip the console window
    pygame.image.save(_screen.screen, './first_direction.BMP')
    G.quit()


def draw_track():
    _param = Parameters()  # Call the variables
    path = np.load('./total_path.npy')
    _screen = Screen()
    _target = Target()
    _cursor = Cursor()
    _event = Event()
    cur_list = []
    color_list = [(255, 0, 0), (125, 10, 10), (10, 255, 10), (10, 10, 255), (10, 125, 10), (10, 10, 125),
                  (125, 125, 10),
                  (125, 10, 125), (10, 125, 125), (125, 125, 125)]

    _screen.overwrite()  # Fill the background with black color.
    for i in range(path.shape[0]):
        cur_list.append([path[i][0], path[i][1]])
    a = 0

    for idx, (cur_x, cur_y) in enumerate(cur_list):
        G.draw.circle(_screen.screen, G.Color(color_list[a]), (cur_x, cur_y), 1.5, 3)
        if idx != 0 and idx % 1500 == 0:
            a += 1
        if a >= 10:
            a -= 10

    flip()
    pygame.image.save(_screen.screen, './total_trajectory.BMP')
    G.quit()


if __name__ == '__main__':
    track_direction()
    draw_track()