from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np
import sys

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

GRID_SIZE = 6
PICK_UP_NUM_OBJECTS_PER = 4

BG_CHAR = ' '
AGENT_CHAR = 'A'
WALL_CHAR = '#'

letters = [chr(i) for i in range(97, 97 + 26)]
base_colours = {
    "red": (1., 0., 0.),
    "green": (0., 1., 0.),
    "blue": (0., 0., 1.),
    "yellow": (1., 1, 0.),
    "purple": (1., 0., 1.),
    "teal": (0., 1., 1.),
}

base_objects = ["square", "diamond"]

objects = {}
i = 0
for o in base_objects:
    for c, c_vec in base_colours.items():
        objects[c + "_" + o] = {"char": letters[i],
                                "color": c_vec} 
        i += 1
COLOURS = {d["char"]: d["color"] for d in objects.values()} 

COLOURS.update({
    BG_CHAR: (0, 0, 0), 
    AGENT_CHAR: (1., 1., 1.),
    WALL_CHAR: (0.5, 0.5, 0.5),
    })

print(COLOURS)

def make_game(game_type, good_color, bad_color, losing=False):
    if game_type == "pick_up":
        shape = "square"
        num_objects = PICK_UP_NUM_OBJECTS_PER
    else:
        raise ValueError("Unknown game type: {}".format(game_type))

    if losing:
        good_color, bad_color = bad_color, good_color  # switcheroo

    good_obj = good_color + "_" + shape 
    bad_obj = bad_color + "_" + shape 

    good_char = objects[good_obj]["char"] 
    bad_char = objects[bad_obj]["char"] 


    grid = []
    grid.append(["#"] * (GRID_SIZE + 2))
    for _ in range(GRID_SIZE):
        grid.append(["#"]  + ([" "] * GRID_SIZE) + ["#"])
    grid.append(["#"] * (GRID_SIZE + 2))

    these_drapes = {good_char: ascii_art.Partial(ValueDrape, value=1.),
                    bad_char: ascii_art.Partial(ValueDrape, value=-1.)}
    locations = [(i, j) for i in range(1, GRID_SIZE + 1) for j in range(1, GRID_SIZE + 1)] 
    np.random.shuffle(locations)
        
    agent_start = locations.pop()
    grid[agent_start[0]][agent_start[1]] = AGENT_CHAR

    for _ in range(num_objects):
        bad_loc = locations.pop()
        grid[bad_loc[0]][bad_loc[1]] = bad_char 
        good_loc = locations.pop()
        grid[good_loc[0]][good_loc[1]] = good_char 

    grid = [''.join(l) for l in grid]  

    return ascii_art.ascii_art_to_game(
        grid, what_lies_beneath=' ',
        sprites={
            'A': PlayerSprite},
        drapes=these_drapes,
        update_schedule=["A"] + list(these_drapes.keys()))


class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):

        if actions == 0:
            self._north(board, the_plot)
        elif actions == 1:
            self._east(board, the_plot)
        elif actions == 2:
            self._south(board, the_plot)
        elif actions == 3:
            self._west(board, the_plot)
        elif actions == 4:
            self._stay(board, the_plot)
        elif actions == 5:
          the_plot.terminate_episode()


class ValueDrape(plab_things.Drape):
    def __init__(self, curtain, character, value):
        super(ValueDrape, self).__init__(curtain, character)
        self._value = value

    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_position = things['A'].position

        if self.curtain[player_position]:
            the_plot.add_reward(self._value)
            self.curtain[player_position] = False
            if self._value > 0 and not self.curtain.any():
                the_plot.terminate_episode()


def curse_color(c):
    return tuple(int(999 * x) for x in c)


def main(argv=()):
    game = make_game("pick_up", "red", "blue")

    scroll_size = GRID_SIZE + 2 if GRID_SIZE % 2 else GRID_SIZE + 1
    cropper = cropping.ScrollingCropper(
        rows=scroll_size, cols=scroll_size, to_track=['A'], pad_char=' ',
        scroll_margins=(None, None))

    print({k: curse_color(v) for k, v in COLOURS.items()})
    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_RIGHT: 1,
                         curses.KEY_DOWN: 2, curses.KEY_LEFT: 3,
                         -1: 4,
                         'q': 5, 'Q': 5},
        delay=100, colour_fg={k: curse_color(v) for k, v in COLOURS.items()},
        colour_bg={AGENT_CHAR: (0, 0, 0)},
        croppers=[cropper])

    ui.play(game)


if __name__ == '__main__':
    main(sys.argv)
