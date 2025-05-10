import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

data = np.load('../Experiment/Data/MV01/RUN01/behav_data.npy')
print(data.shape)
print(data[:, 0], data[:, 1]) # Cursor_x, Cursor_y
print(data[:, 2], data[:, 3]) # Target_x, Target_y

import pygame as G
from pygame.rect import *
print(data[1][2])
G.init()
screen = G.display.set_mode([1440, 850])  # Set the console window size.
frame = []

for i in range(data[:, 0].shape[0]):
    screen.fill(G.Color(0, 0, 0))
    c_rect = Rect(int(data[i, 0]), int(data[i, 1]), 20, 20)
    G.draw.ellipse(screen, G.Color(128, 128, 128), c_rect, width=3)
    t_rect = Rect(int(data[i, 2]), int(data[i, 3]), 70, 70)
    G.draw.ellipse(screen, G.Color(128, 128, 128), t_rect, 1)
    if data[i, 6]:
        G.draw.ellipse(screen, G.Color(255, 0, 0), t_rect, 0)
    G.display.flip()
    G.image.save(screen, 'abc.BMP')
    png = Image.open('./abc.BMP')
    png.load() # required for png.split()

    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])
    frame.append(background)

frame_array = []
for i in range(len(frame)):
    frame_array.append(np.array(frame[i]))

height, width, layers = frame_array[0].shape
size = (width, height)

out = cv2.VideoWriter('./video.mp4', fourcc=0x7634706d, fps=60, frameSize=size)
for i in range(data[:, 0].shape[0]):
    out.write(frame_array[i])
out.release()