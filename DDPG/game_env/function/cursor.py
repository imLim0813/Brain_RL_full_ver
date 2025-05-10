import pygame as G
import numpy as np
from DDPG.game_env.function.parameters import Parameters
from pygame.rect import *


class Cursor:
    def __init__(self):
        self._param = Parameters()
        self.cur_x = 960
        self.cur_y = 540
        self.max_x = self._param.width - self._param.cursor_diameter
        self.max_y = self._param.height - self._param.cursor_diameter

    def update(self, screen):
        G.draw.line(screen.screen, self._param.white, (self.cur_x - 10, self.cur_y), (self.cur_x + 10, self.cur_y), 3)
        G.draw.line(screen.screen, self._param.white, (self.cur_x, self.cur_y - 10), (self.cur_x, self.cur_y + 10), 3)

    def move(self, mode, action):

        # (r, theta) to (x, y)
        act_x = action[0] * np.cos(np.deg2rad((action[1])))
        act_y = action[0] * np.sin(np.deg2rad(action[1]))

        if mode == 'base':
            self.cur_x, self.cur_y = base(self.cur_x, self.cur_y, act_x, act_y, self.max_x, self.max_y)
        elif mode == 'adapt':
            self.cur_x, self.cur_y = adapt(self.cur_x, self.cur_y, act_x, act_y, self.max_x, self.max_y)
        elif mode == 'reverse':
            self.cur_x, self.cur_y = reverse(self.cur_x, self.cur_y, act_x, act_y, self.max_x, self.max_y)
        else:
            print('Choose the mode among...')
            RuntimeError()

        return [self.cur_x, self.cur_y] # If you can, list to tuple


def base(cur_x, cur_y, act_x, act_y, max_x, max_y):

    cur_x += act_x * 2
    cur_y += act_y * 2

    if cur_y <= 0:
        cur_y = 0
    elif cur_y >= max_y:
        cur_y = max_y

    if cur_x <= 0:
        cur_x = 0
    elif cur_x >= max_x:
        cur_x = max_x

    return cur_x, cur_y


def adapt(cur_x, cur_y, act_x, act_y, max_x, max_y, degree=-90):

    prev_x = act_x * 2
    prev_y = act_y * 2

    if prev_x > 0:
        theta_final = degree + np.rad2deg(np.arctan(prev_y / prev_x))
    elif prev_x < 0:
        theta_final = 180 + degree + np.rad2deg(np.arctan(prev_y / prev_x))

    prev_x_rot = np.sqrt(prev_x ** 2 + prev_y ** 2) * np.cos(np.deg2rad(theta_final))
    prev_y_rot = np.sqrt(prev_x ** 2 + prev_y ** 2) * np.sin(np.deg2rad(theta_final))

    cur_x += prev_x_rot
    cur_y += prev_y_rot

    if cur_y <= 0:
        cur_y = 0
    elif cur_y >= max_y:
        cur_y = max_y

    if cur_x <= 0:
        cur_x = 0
    elif cur_x >= max_x:
        cur_x = max_x

    return cur_x, cur_y


def reverse(cur_x, cur_y, act_x, act_y, max_x, max_y, degree=0):
    prev_x = act_x * 2
    prev_y = act_y * 2

    cur_x -= prev_y
    cur_y -= prev_x

    if cur_y <= 0:
        cur_y = 0
    elif cur_y >= max_y:
        cur_y = max_y

    if cur_x <= 0:
        cur_x = 0
    elif cur_x >= max_x:
        cur_x = max_x

    return cur_x, cur_y

