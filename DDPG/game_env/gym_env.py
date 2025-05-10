from abc import ABC
import gym
import pygame as G
import cv2

from game_env.function.utils import *
from game_env.function.parameters import Parameters
from game_env.function.screen import Screen, flip
from game_env.function.target import Target
from game_env.function.event import Event
from game_env.function.cursor import Cursor

path = np.load('../../Path/total_path.npy').astype('int')


class Denovo_DDPG(gym.Env, ABC):
    def __init__(self):
        # Call the library and set the base.

        self.parameter = Parameters()
        self.clock = pg.time.Clock()
        self.screen = Screen()
        self.target = Target()
        self.cursor = Cursor()
        self.event = Event()

        visible_mouse(False)

        self.done = False
        self.count = 0

        # Action
        act_high = 1.0
        self.action_r = gym.spaces.Box(low=np.float(0), high=np.float(act_high), shape=(1,))

        self.action_theta = gym.spaces.Box(low=np.float(-act_high), high=np.float(act_high), shape=(1,))

        # Observation
        obs_high = np.array([1920., 1080.], dtype=np.float)
        obs_low = np.array([0., 0.], dtype=np.float)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(2,), dtype=np.float)

        # State
        self.state = np.array([self.parameter.init_x, self.parameter.init_y,
                               self.parameter.init_x, self.parameter.init_y], dtype=np.float)

    def step(self, r, theta):
        # Move target position
        self.target.move(path=path, idx=self.count)

        # Move cursor
        action = np.array([r[0], theta[0]])
        tmp = self.cursor.move('base', action)

        # Update state
        self.state = np.array([tmp[0], tmp[1], self.target.pos()[0], self.target.pos()[1]], dtype=np.float)

        # Set reward ( inverse gaussian )
        dt = self.distance()
        reward = dist_reward(dt, sigma=100)

        # Done
        if self.parameter.duration < self.count:
            self.done = True
        if dt > 500:
            self.done = True

        # Info
        info = {}

        self.count += 1

        # If self.count % 1499 == 0,
        if self.count != 0 and self.count % 1500 == 0:
            self.cursor = Cursor()
            # Reset Cursor ( because it doesn't follow the path )

        # the number of path's index : 30000
        if self.count == 30000:
            self.done = True

        return self.state, reward, self.done, info

    def reset(self):
        # Reset count
        self.count = 0

        # done : False
        self.done = False
        # self.screen = Screen()

        # Reset target
        self.target = Target()

        # Reset Cursor
        self.cursor = Cursor()
        self.event = Event()

        # Reset state
        self.state = np.array([self.parameter.init_x, self.parameter.init_y,
                               self.parameter.init_x, self.parameter.init_y], dtype=np.float)

        return self.state

    def render(self):

        # If don't use this, the console window is not the same with the state.
        pg.event.pump()

        # Fill the window with black ground.
        self.screen.overwrite()

        # Display target.
        self.target.update('base', self.screen)

        # If hit, then the target will show red color.
        if self.hit():
            self.event.hit_target(self.screen, self.target)
        else:
            pass

        # Display cursor
        self.cursor.update(self.screen)

        # Set the hertz
        clock_tick(self.parameter.hertz)

        # Update the console window.
        flip()

    def to_frame(self):

        string_image = G.image.tostring(self.screen.screen, 'RGB')
        temp_surf = G.image.fromstring(string_image, (self.parameter.width, self.parameter.height), 'RGB')
        tmp_arr = G.surfarray.array3d(temp_surf)
        tmp_arr = tmp_arr.transpose((1, 0, 2))

        # 이미지 사이즈 조정.
        # 이 실험 환경에서는 모든 공간에 타겟과 커서가 움직이므로 crop은 하지 않음.

        image = cv2.resize(tmp_arr, (227, 227), interpolation=cv2.INTER_AREA)

        # 토치 환경에서는 (배치 사이즈, 채널, height, width)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, -1)
        return image

    def hit(self):
        # If cursor is in the target area, then 'hit'.
        if self.state[2] <= self.state[0] <= self.state[2] + self.parameter.target_diameter \
                and self.state[3] <= self.state[1] <= self.state[3] + self.parameter.target_diameter:
            return True
        else:
            return False

    def distance(self):
        # Euclidean distance
        tc_x = (self.state[2] * 2 + self.parameter.target_diameter) // 2
        tc_y = (self.state[3] * 2 + self.parameter.target_diameter) // 2
        cc_x = (self.state[0] * 2 + self.parameter.cursor_diameter) // 2
        cc_y = (self.state[1] * 2 + self.parameter.cursor_diameter) // 2
        return np.sqrt((tc_x - cc_x) ** 2 + (tc_y - cc_y) ** 2)


def dist_reward(distance, sigma):
    # Inverse Gaussian
    return np.exp(-((distance ** 2) / (2 * (sigma ** 2))))


def dt_state(state):
    tmp = np.array([0, 0], dtype=np.float32)

    dt_x = state[0] - state[2]
    dt_y = state[1] - state[3]

    tmp[0] = dt_x
    tmp[1] = dt_y

    return tmp

