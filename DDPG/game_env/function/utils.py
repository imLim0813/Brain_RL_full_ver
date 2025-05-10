import pygame as pg
import numpy as np


def visible_mouse(a=True):
    pg.mouse.set_visible(a)


def clock_tick(time):
    clock = pg.time.Clock()
    clock.tick(time)

