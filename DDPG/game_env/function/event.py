from DDPG.game_env.function.parameters import Parameters
import pygame as G


class Event:

    def __init__(self):
        self._param = Parameters()

    def hit_target(self, screen, target):
        G.draw.ellipse(screen.screen, self._param.red, target.rect, 0)
