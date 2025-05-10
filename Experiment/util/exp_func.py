import pygame as G
import numpy as np
import cv2
import os

from PIL import Image
from pygame.rect import *


class Parameters:
    def __init__(self, mode='base'):
        # Color Parameters
        self.red = G.Color(255, 0, 0)  # When the cursor hit the target
        self.gray = G.Color(86, 86, 86)  # Cursor, Target : Base perturbation
        self.black = G.Color(0, 0, 0)  # Screen Background
        self.blue = G.Color(0, 0, 255)  # Target : Adaptation perturbation
        self.yellow = G.Color(255, 255, 0)  # Target : Denovo perturbation
        # # Supplementary color, just in case
        self.green = G.Color(0, 255, 0)
        self.white = G.Color(255, 255, 255)

        # Display Parameters
        self.width = 1920
        self.height = 1080
        self.cursor_diameter = 20
        self.target_diameter = 70

        # Time Parameters
        self.hertz = 30  # FPS
        self.time = 25  # Time of one trial
        self.trial = 20
        self.duration = self.hertz * self.time  # the number of frame.

        # Control Parameters
        self.count = 0
        self.index = 0
        self.mode = mode

        # Target Parameters
        self.target_speed = [2, 2]  # Initialize target's speed.
        self.target_x = 960  # Initialize target's x position.
        self.target_y = 540  # Initialize target's y position.


class Screen:
    def __init__(self):
        G.init()  # Initiate Pygame.
        self._param = Parameters()
        # self.screen = G.display.set_mode([self._param.width, self._param.height])  # Set the console window size.
        self.screen = G.display.set_mode((0, 0), G.FULLSCREEN)

    def overwrite(self):
        self.screen.fill(self._param.black)  # Fill the window background with black color.


class Cursor:
    def __init__(self, speed=3.5):
        self._param = Parameters()
        self.cur_x = 995  # Initialize the cursor's x position.
        self.cur_y = 575  # Initialize the cursor's y position.
        self.max_x = self._param.width - self._param.cursor_diameter  # Define the max value of cursor's x position.
        self.max_y = self._param.height - self._param.cursor_diameter  # Define the max value of cursor's y position.
        self._speed = speed

    def update(self, cur_x, cur_y, _screen):
        G.draw.line(_screen.screen, self._param.white, (cur_x-10, cur_y), (cur_x+10, cur_y),  width=3)
        G.draw.line(_screen.screen, self._param.white, (cur_x, cur_y-10), (cur_x, cur_y+10),  width=3)

    def mode(self, _screen, _target, _joystick):
        if _joystick == 'None':
            prev_x = 0
            prev_y = 0

            self.cur_x += prev_x
            self.cur_y += prev_y

            return int(self.cur_x), int(self.cur_y), prev_x, prev_y

        if self._param.mode == 'base':

            # 1) get the joystick inputs.
            prev_x = _joystick.joystick.get_axis(0) * self._speed
            prev_y = _joystick.joystick.get_axis(1) * self._speed

            # 2) Add the joystick inputs with existing positions.
            self.cur_x += prev_x
            self.cur_y += prev_y

            # Confine the range of cursor's y position.
            if self.cur_y <= 10:
                self.cur_y = 10
            elif self.cur_y >= self.max_y:
                self.cur_y = self.max_y

            # Confine the range of cursor's y position.
            if self.cur_x <= 10:
                self.cur_x = 10
            elif self.cur_x >= self.max_x:
                self.cur_x = self.max_x

            # The console shows the cursor only with integers. So, we don't have to record the floats.
            return int(self.cur_x), int(self.cur_y), prev_x, prev_y

        elif self._param.mode == 'adaptation':
            pass

        elif self._param.mode == 'denovo':
            pass

        else:
            RuntimeError('You select wrong mode!\n Please chooses among ["base", "adaptation", "denovo"]')

        return False


class Target:
    def __init__(self):
        self._param = Parameters()
        self.rect = Rect(self._param.target_x, self._param.target_y,
                         self._param.target_diameter, self._param.target_diameter)
        self.target = None
        self.idx = 0
        self.hit = 0

    def move(self, path, idx):
        self.rect = Rect(path[idx][0], path[idx][1],
                         self._param.target_diameter, self._param.target_diameter)

        # Actually, below conditional codes don't need because I already confine
        # the limit of target's position in 'Path.py'. Just in case, I didn't erase these codes.
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self._param.width:
            self.rect.right = self._param.width
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > self._param.height:
            self.rect.bottom = self._param.height

    def update(self, _screen, cursor_x, cursor_y):
        if self._param.mode == 'base':
            self.target = G.draw.ellipse(_screen.screen, self._param.gray,
                                         self.rect, 0)

            if self.rect[0] <= cursor_x <= self.rect[0] + self.rect[2] \
                    and self.rect[1] <= cursor_y <= self.rect[1] + self.rect[3]:
                self.target = G.draw.ellipse(_screen.screen, self._param.red,
                                             self.rect, 0)
                self.hit = 1
            else:
                self.hit = 0

        elif self._param.mode == 'adaptation':
            self.target = G.draw.ellipse(_screen.screen, self._param.blue,
                                         self.rect, 0)
        elif self._param.mode == 'denovo':
            self.target = G.draw.ellipse(_screen.screen, self._param.yellow,
                                         self.rect, 0)
        else:
            RuntimeError('You select wrong mode!\n Please chooses among ["base", "adaptation", "denovo"]')

        return self.hit

    def get_pos(self):
        return self.rect.copy()


class Joystick:
    def __init__(self):
        G.joystick.init()
        self.joystick = G.joystick.Joystick(0)
        self.joystick.init()

    def __str__(self):
        return 'Axis x : {:.3f}, Axis y : {:.3f}'.format(self.joystick.get_axis(0), self.joystick.get_axis(1))


class Event:
    def __init__(self):
        self._param = Parameters()

    def hit_target(self, _screen, _target):
        G.draw.ellipse(_screen.screen, self._param.red, _target.rect, 0)


class Text:
    def __init__(self):
        G.font.init()
        self.text = G.font.SysFont('AppleGothic', 50, True, False)
        self._param = Parameters()

    def wait(self):
        wait_1 = self.text.render('Wait for signal', True, self._param.gray)
        wait_2 = wait_1.get_rect()
        wait_2.x = 800
        wait_2.y = 450

        return wait_1, wait_2

    def thank(self):
        thank_1 = self.text.render('Thank you', True, self._param.gray)
        thank_2 = thank_1.get_rect()
        thank_2.x = 800
        thank_2.y = 450

        return thank_1, thank_2

    def cross(self):
        cross_1 = self.text.render('+', True, self._param.white)
        cross_2 = cross_1.get_rect()
        cross_2.x = 960
        cross_2.y = 540

        return cross_1, cross_2


class directory:
    def __init__(self):
        self.dir = './exp_result/'
        self.subj = input('Please Enter the Subject ID : ')

    def mkdir(self):
        if not os.path.exists(self.dir + self.subj):
            os.makedirs(self.dir + self.subj)


def flip():
    G.display.flip()


def visible_mouse(tmp=True):
    G.mouse.set_visible(tmp)

