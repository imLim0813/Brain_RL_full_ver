import os
import numpy as np
import cv2
import pygame as G
import matplotlib.pyplot as plt

from collections import deque
from PIL import Image
from pygame.rect import *
from game_env.function.parameters import *

global _param
_param = Parameters()


def to_frame(state):
    cur_x = state[0]
    cur_y = state[1]
    target_x = state[2]
    target_y = state[3]

    screen = G.display.set_mode([_param.width, _param.height])
    frame = None

    screen.fill(_param.black)

    G.draw.line(screen, _param.white, (cur_x - 10, cur_y), (cur_x + 10, cur_y), 3)
    G.draw.line(screen, _param.white, (cur_x, cur_y - 10), (cur_x, cur_y + 10), 3)

    target_rect = Rect(target_x, target_y, _param.target_diameter, _param.target_diameter)
    G.draw.ellipse(screen, _param.white, target_rect, 0)

    if target_x <= cur_x <= target_x + _param.target_diameter \
            and target_y <= cur_y <= target_y + _param.target_diameter:
        G.draw.ellipse(screen, _param.red, target_rect, 0)

    string_image = G.image.tostring(screen, 'RGB')
    temp_surf = G.image.fromstring(string_image, (_param.width, _param.height), 'RGB')
    tmp_arr = G.surfarray.array3d(temp_surf)
    tmp_arr = tmp_arr.transpose((1, 0, 2))

    # 이미지 사이즈 조정.
    # 이 실험 환경에서는 모든 공간에 타겟과 커서가 움직이므로 crop은 하지 않음.
    
    image = cv2.resize(tmp_arr, (227, 227), interpolation=cv2.INTER_AREA)

    # 토치 환경에서는 (배치 사이즈, 채널, height, width)
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, -1)
    return image

