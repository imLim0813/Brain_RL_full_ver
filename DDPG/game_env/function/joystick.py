import pygame as G


class Joystick:

    def __init__(self):
        G.joystick.init()
        self.joystick = G.joystick.Joystick(0)
        self.joystick.init()
        print("Joystick enable : ", self.joystick.get_init())

    def __str__(self):
        return 'Axis x : {:.3f}, Axis y : {:.3f}'.format(self.joystick.get_axis(0), self.joystick.get_axis(1))
