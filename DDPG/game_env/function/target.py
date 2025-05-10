from pygame.rect import *
from DDPG.game_env.function.parameters import Parameters

import pygame as pg


class Target:

    def __init__(self):
        self._param = Parameters()

        self.rect = Rect(self._param.init_x,
                         self._param.init_y,
                         self._param.target_diameter,
                         self._param.target_diameter)
        self.target = None
        self.idx = 0

    def move(self, path, idx):
        
        self.rect = Rect(path[idx][0], path[idx][1],
                         self._param.target_diameter, self._param.target_diameter)

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self._param.width:
            self.rect.right = self._param.width
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > self._param.height:
            self.rect.bottom = self._param.height

    def update(self, mode, screen):
        if mode == 'base':
            self.target = pg.draw.ellipse(screen.screen, self._param.white, self.rect, 3)
        elif mode == 'adapt':
            self.target = pg.draw.ellipse(screen.screen, self._param.blue, self.rect, 3)
        elif mode == 'reverse':
            self.target = pg.draw.ellipse(screen.screen, self._param.yellow, self.rect, 3)

    def pos(self):
        return [list(self.rect.copy())[0], list(self.rect.copy())[1]]
