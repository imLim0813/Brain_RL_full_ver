import time
import pickle
import sys

import pygame.time
import pymongo.pool

from util.exp_func import *

_param = Parameters(mode='base')  # Call the variables
_text = Text()  # Call the Text class
wait_1, wait_2 = _text.wait()
thank_1, thank_2 = _text.thank()
cross_1, cross_2 = _text.cross()

_event = Event()
_dir = directory()
_dir.mkdir()

path_list = [np.load('../Path/Path_{}.npy'.format(i)) for i in range(1, 21, 1)]


def main():
    _screen = Screen()
    G.mouse.set_visible(False)

    n_run = 0

    print(
        '''\nDenovo Game_ver : 3.0\n
        Please Notice that you should use lower case of alphabet (i.e. 'r', 's') to start the game.
        \nThank you!\n''')

    # module to synchronize.
    while n_run != 6:

        _screen.screen.blit(wait_1, wait_2)
        flip()

        for event in G.event.get():
            if event.type == G.QUIT:
                G.quit()
                sys.exit()
        keys = G.key.get_pressed()
        if keys[G.K_r]:
            while True:
                for event in G.event.get():
                    if event.type == G.QUIT:
                        G.quit()
                        sys.exit()

                keys = G.key.get_pressed()
                if keys[G.K_s]:
                    break
                else:
                    continue
        else:
            continue

        _trial = 0
        record = {'cursor': [], 'target': [], 'joystick': [], 'hit': [], 'time': []}
        start_record = time.time()

        while _trial != 20:  # total time : 20 seconds(trial) * (30 seconds(Game) + 5 seconds(Wait))

            path = path_list[_trial]
            _target = Target()
            _cursor = Cursor()
            _joystick = Joystick()
            clock = pygame.time.Clock()

            for i in range(_param.duration):  # play the game 25 seconds. ( 1500 Frames : 60 FPS * 25 seconds )

                G.event.pump()
                _screen.overwrite()  # Fill the background with black color.
                _target.move(path=path, idx=i)  # Move the target based on path.

                cursor_x, cursor_y, stick_x, stick_y = _cursor.mode(_screen, _target, _joystick=_joystick)  # Move the cursor

                hit = _target.update(_screen, cursor_x, cursor_y)  # Display the moved target

                _cursor.update(cursor_x, cursor_y, _screen)  # Display the moved cursor.

                clock.tick(60)
                flip()  # Flip the console window

                record['cursor'].append([cursor_x, cursor_y])
                record['target'].append([_target.get_pos()[0], _target.get_pos()[1]])
                record['joystick'].append([stick_x, stick_y])
                record['hit'].append([hit])
                record['time'].append([time.time() - start_record])

            _trial += 1

            # Display cursor and target.
            _screen.overwrite()
            _target = Target()
            _cursor = Cursor()
            _target.update(_screen, cursor_x, cursor_y)  # Display the moved target
            _cursor.update(_cursor.cur_x, _cursor.cur_y, _screen)  # Display the moved cursor.
            flip()

            G.time.delay(10100)

            print(time.time() - start_record)

        # save the behavior data.
        with open(_dir.dir + _dir.subj + f'/behavior_data_{n_run + 1}.pkl', 'wb') as f:
            pickle.dump(record, f)

        while n_run == 2:
            _screen.overwrite()
            _screen.screen.blit(cross_1, cross_2)
            flip()
            for event in G.event.get():
                if event.type == G.QUIT:
                    G.quit()
                    sys.exit()

            keys = G.key.get_pressed()
            if keys[G.K_c]:
                break
            else:
                continue

        n_run += 1

    # Display Thank text
    _screen.overwrite()
    _screen.screen.blit(thank_1, thank_2)
    flip()

    while True:
        for event in G.event.get():
            if event.type == G.QUIT:
                G.quit()
                sys.exit()

        keys = G.key.get_pressed()
        if keys[G.K_q]:
            break
        else:
            continue


if __name__ == '__main__':
    main()
