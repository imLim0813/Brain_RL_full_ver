import pygame as G
from DDPG.game_env.function.parameters import Parameters


def flip():
    G.display.flip()


class Screen:

    def __init__(self):
        G.init()
        self._param = Parameters()
        self.screen = G.display.set_mode([self._param.width, self._param.height])

    def overwrite(self):
        self.screen.fill(self._param.black)

