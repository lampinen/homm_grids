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

GRID_SIZE = 6  # should be even

PICK_UP_NUM_OBJECTS_PER = 4

SEQ_IMIT_MOVE_LIMIT = 8
SEQ_IMIT_VALUE_PER = 0.5

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
    if losing:
        good_color, bad_color = bad_color, good_color  # switcheroo

    these_sprites = {}
    these_drapes = {}
    if game_type == "pick_up":
        shape = "square"
        num_objects = PICK_UP_NUM_OBJECTS_PER
        good_obj = good_color + "_" + shape 
        bad_obj = bad_color + "_" + shape 

        good_char = objects[good_obj]["char"] 
        bad_char = objects[bad_obj]["char"] 

        these_drapes.update(
            {good_char: ascii_art.Partial(ValueDrape, value=1.),
             bad_char: ascii_art.Partial(ValueDrape, value=-1.)})
        update_schedule = ["A", good_char, bad_char]
    elif game_type == "sequence_imitation":
        shape = "square"
        num_objects = 1
        good_obj = good_color + "_" + shape 
        bad_obj = bad_color + "_" + shape 

        good_char = objects[good_obj]["char"] 
        bad_char = objects[bad_obj]["char"] 

        these_sprites.update(
            {good_char: ascii_art.Partial(DancerSprite, 
                                          value=SEQ_IMIT_VALUE_PER),
             bad_char: ascii_art.Partial(DancerSprite,
                                         value=-SEQ_IMIT_VALUE_PER)})
        update_schedule = ["A", good_char, bad_char]

    else:
        raise ValueError("Unknown game type: {}".format(game_type))

    grid = []
    grid.append(["#"] * (GRID_SIZE + 2))
    for _ in range(GRID_SIZE):
        grid.append(["#"]  + ([" "] * GRID_SIZE) + ["#"])
    grid.append(["#"] * (GRID_SIZE + 2))
    if game_type in ["pick_up"]:
        locations = [(i, j) for i in range(1, GRID_SIZE + 1) for j in range(1, GRID_SIZE + 1)] 
    elif game_type == "sequence_imitation":
        quarts = [1 + GRID_SIZE // 4, GRID_SIZE - GRID_SIZE // 4]
        locations = [(x, y) for x in quarts for y in quarts]

    np.random.shuffle(locations)
    for _ in range(num_objects):
        bad_loc = locations.pop()
        grid[bad_loc[0]][bad_loc[1]] = bad_char 
        good_loc = locations.pop()
        grid[good_loc[0]][good_loc[1]] = good_char 
    agent_start = locations.pop()
    grid[agent_start[0]][agent_start[1]] = AGENT_CHAR
    these_sprites.update({'A': ascii_art.Partial(PlayerSprite, game_type=game_type)})

    grid = [''.join(l) for l in grid]  

    game = ascii_art.ascii_art_to_game(
        grid,
        what_lies_beneath=' ',
        sprites=these_sprites,
        drapes=these_drapes,
        update_schedule=update_schedule)

    if game_type == "pick_up":
        game.the_plot["num_picked_up"] = 0
    elif game_type == "sequence_imitation":
        game.the_plot["good_move"] = -1  # no valid move on first turn
        game.the_plot["bad_move"] = -1
    return game


class PlayerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character, game_type):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable='#')
        self.game_type = game_type

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

        if self.game_type == "sequence_imitation":
            if actions == the_plot["good_move"]:
                the_plot.add_reward(the_plot["good_value"])
            elif actions == the_plot["bad_move"]:
                the_plot.add_reward(the_plot["bad_value"])


class DancerSprite(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character, value):
        super(DancerSprite, self).__init__(
        corner, position, character, impassable='#')
        self.value = value
        self.offset_x = 0
        self.offset_y = 0
        self.max_offset = GRID_SIZE // 4
        self.num_moves = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if self.num_moves >= SEQ_IMIT_MOVE_LIMIT or self.position == things[AGENT_CHAR].position: 
            the_plot.terminate_episode()
        row, col = self.position
        max_offset = self.max_offset
        
        possible_moves = []
        if self.offset_x < max_offset:
            possible_moves.append(1)
        if self.offset_x > -max_offset:
            possible_moves.append(3)
        if self.offset_y < max_offset:
            possible_moves.append(2)
        if self.offset_y > -max_offset:
            possible_moves.append(0)

        if self.value > 0:
            move = np.random.choice(possible_moves) 
            the_plot["good_move"] = move
            the_plot["good_value"] = self.value
        else:
            good_move = the_plot["good_move"]
            possible_moves = [x for x in possible_moves if x != good_move] 
            move = np.random.choice(possible_moves) 
            the_plot["bad_move"] = move
            the_plot["bad_value"] = self.value

        if move == 0:
            self._north(board, the_plot)
            self.offset_y -= 1
        elif move == 1:
            self._east(board, the_plot)
            self.offset_x += 1
        elif move == 2:
            self._south(board, the_plot)
            self.offset_y += 1
        elif move == 3:
            self._west(board, the_plot)
            self.offset_x -= 1
        self.num_moves += 1



class ValueDrape(plab_things.Drape):
    def __init__(self, curtain, character, value):
        super(ValueDrape, self).__init__(curtain, character)
        self._value = value

    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_position = things['A'].position

        if self.curtain[player_position]:
            the_plot.add_reward(self._value)
            self.curtain[player_position] = False
            the_plot["num_picked_up"] += 1
            if the_plot["num_picked_up"] == PICK_UP_NUM_OBJECTS_PER:
                the_plot.terminate_episode()


def curse_color(c):
    return tuple(int(999 * x) for x in c)


def main(argv=()):

    scroll_size = 2 * GRID_SIZE + 1
    cropper = cropping.ScrollingCropper(
        rows=scroll_size, cols=scroll_size, to_track=['A'], pad_char=' ',
        scroll_margins=(None, None))

    for game_type in ["sequence_imitation", "pick_up"]:
        game = make_game(game_type, "red", "blue")

        ui = human_ui.CursesUi(
            keys_to_actions={curses.KEY_UP: 0, curses.KEY_RIGHT: 1,
                             curses.KEY_DOWN: 2, curses.KEY_LEFT: 3,
                             -1: 4,
                             'q': 5, 'Q': 5},
            delay=None, colour_fg={k: curse_color(v) for k, v in COLOURS.items()},
            colour_bg={AGENT_CHAR: (0, 0, 0)},
            croppers=[cropper])

        ui.play(game)


if __name__ == '__main__':
    main(sys.argv)
