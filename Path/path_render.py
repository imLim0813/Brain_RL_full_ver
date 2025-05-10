from DDPG.game_env.function.cursor import *
from DDPG.game_env.function.screen import *
from DDPG.game_env.function.target import *
from DDPG.game_env.function.event import *
from DDPG.game_env.function.utils import *

_param = Parameters()  # Call the variables

path = np.load('./total_path.npy')

clock_tick(_param.hertz)
_screen = Screen()
_target = Target()
_cursor = Cursor()
_event = Event()
cur_list = []
color_list = [(255, 0, 0), (125, 10, 10), (10, 255, 10), (10, 10, 255), (10, 125, 10), (10, 10, 125), (125, 125, 10),
              (125, 10, 125), (10, 125, 125), (125, 125, 125)]

for i in range(_param.duration):
    G.event.pump()
    _screen.overwrite()  # Fill the background with black color.
    _target.move(path=path, idx=i)  # Move the target based on path.
    cur_list.append([path[i][0], path[i][1]])

    a = 0
    for idx, (cur_x, cur_y) in enumerate(cur_list):
        G.draw.circle(_screen.screen, G.Color(color_list[a]), (cur_x + 35, cur_y + 35), 1.5, 3)
        if idx != 0 and idx % 1500 == 0:
            a += 1
        if a >= 10:
            a -= 10
    target_pos = list(_target.pos())

    _target.update(mode='base', screen=_screen)  # Display the moved target
    flip() # Flip the console window