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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

GRID_SIZE = 6  # should be even
SCROLL_SIZE = 2 * GRID_SIZE + 1
UPSAMPLE_SIZE = 7

PICK_UP_NUM_OBJECTS_PER = 4
PUSHER_NUM_OBJECTS_PER = 4
SHOOTER_NUM_OBJECTS_PER = 4

SEQ_IMIT_MOVE_LIMIT = 8
SEQ_IMIT_VALUE_PER = 0.5

NEG_VALUE = -1.

BG_CHAR = ' '
AGENT_CHAR = 'A'
WALL_CHAR = '#'

letters = [chr(i) for i in list(range(97, 97 + 26)) + list(range(66, 66 + 25))] # omit 'A', used for agent

BASE_COLOURS = {
    "red": (1., 0., 0.),
    "green": (0., 1., 0.),
    "blue": (0., 0., 1.),
    "yellow": (1., 1, 0.),
    "pink": (1., 0.3, 1.),
    "cyan": (0., 1., 1.),
    "purple": (0.5, 0., 0.6),
    "ocean": (0.1, 0.4, 0.5),
    "orange": (1., 0.6, 0.),
    "forest": (0., 0.5, 0.),
}

BASE_SHAPES = ["square", "diamond", "triangle", "tee"]

AGENT_SHAPE = "triangle"

OBJECTS = {}
i = 0
for o in BASE_SHAPES:
    for c, c_vec in BASE_COLOURS.items():
        OBJECTS[c + "_" + o] = {"char": letters[i],
                                "color": c_vec} 
        i += 1
COLOURS = {d["char"]: d["color"] for d in OBJECTS.values()} 

COLOURS.update({
    BG_CHAR: (0, 0, 0), 
    AGENT_CHAR: (1., 1., 1.),
    WALL_CHAR: (0.5, 0.5, 0.5),
    })

def make_game(game_type, good_color, bad_color, switched_colors=False):
    if switched_colors:
        good_color, bad_color = bad_color, good_color  # switcheroo

    these_sprites = {}
    these_drapes = {}
    if game_type == "pick_up":
        shape = "square"
        num_objects = PICK_UP_NUM_OBJECTS_PER
        good_obj = good_color + "_" + shape 
        bad_obj = bad_color + "_" + shape 

        good_char = OBJECTS[good_obj]["char"] 
        bad_char = OBJECTS[bad_obj]["char"] 

        these_drapes.update(
            {good_char: ascii_art.Partial(ValueDrape, value=1.),
             bad_char: ascii_art.Partial(ValueDrape, value=NEG_VALUE)})
    elif game_type == "pusher":
        shape = "tee"
        num_objects = PUSHER_NUM_OBJECTS_PER
        good_obj = good_color + "_" + shape 
        bad_obj = bad_color + "_" + shape 

        good_char = OBJECTS[good_obj]["char"] 
        bad_char = OBJECTS[bad_obj]["char"] 

        these_drapes.update(
            {good_char: ascii_art.Partial(PushableDrape, 
                                          value=1.),
             bad_char: ascii_art.Partial(PushableDrape,
                                         value=NEG_VALUE)})
    elif game_type == "shooter":
        shape = "diamond"
        num_objects = SHOOTER_NUM_OBJECTS_PER
        good_obj = good_color + "_" + shape 
        bad_obj = bad_color + "_" + shape 

        good_char = OBJECTS[good_obj]["char"] 
        bad_char = OBJECTS[bad_obj]["char"] 

        these_drapes.update(
            {good_char: ascii_art.Partial(ShootableDrape, value=1.),
             bad_char: ascii_art.Partial(ShootableDrape, value=NEG_VALUE)})
    elif game_type == "sequence_imitation":
        shape = "triangle"
        num_objects = 1
        good_obj = good_color + "_" + shape 
        bad_obj = bad_color + "_" + shape 

        good_char = OBJECTS[good_obj]["char"] 
        bad_char = OBJECTS[bad_obj]["char"] 

        these_sprites.update(
            {good_char: ascii_art.Partial(DancerSprite, 
                                          value=SEQ_IMIT_VALUE_PER),
             bad_char: ascii_art.Partial(DancerSprite,
                                         value=SEQ_IMIT_VALUE_PER * NEG_VALUE)})
    else:
        raise ValueError("Unknown game type: {}".format(game_type))

    update_schedule = [AGENT_CHAR, good_char, bad_char]
    if game_type == "pusher":
        update_schedule = [good_char, bad_char, AGENT_CHAR]

    grid = []
    grid.append(["#"] * (GRID_SIZE + 2))
    for _ in range(GRID_SIZE):
        grid.append(["#"]  + ([" "] * GRID_SIZE) + ["#"])
    grid.append(["#"] * (GRID_SIZE + 2))
    if game_type in ["pick_up", "pusher", "shooter"]:
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
    elif game_type == "pusher":
        game.the_plot["num_pushed_off"] = 0
        game.the_plot["player_blocked"] = False
    elif game_type == "shooter":
        game.the_plot["num_shot"] = 0
        game.the_plot["heading"] = 0
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
        if actions is None:  # dummy step at start of game
            return
        elif actions == 8:
          the_plot.terminate_episode()
        elif self.game_type == "shooter" and actions >= 4:
            self._shoot(actions - 4, things, the_plot)
        elif self.game_type == "pusher" and the_plot["player_blocked"]:
            self._stay(board, the_plot)
        else:
            if actions == 0:
                self._north(board, the_plot)
            elif actions == 1:
                self._east(board, the_plot)
            elif actions == 2:
                self._south(board, the_plot)
            elif actions == 3:
                self._west(board, the_plot)
            elif actions >= 4:
                self._stay(board, the_plot)

        if self.game_type == "sequence_imitation":
            if actions == the_plot["good_move"]:
                the_plot.add_reward(the_plot["good_value"])
            elif actions == the_plot["bad_move"]:
                the_plot.add_reward(the_plot["bad_value"])

        the_plot["player_blocked"] = False  # reset for next turn

    def _shoot(self, heading, things, the_plot):
        assert(self.game_type == "shooter")  # TODO: remove when testing is done
        drapes = [v for (k, v) in things.items() if k != self.character]
        pos_x, pos_y = self.position 
        done = False
        while (not done) and (0 < pos_x < GRID_SIZE + 1) and (0 < pos_y < GRID_SIZE + 1):
            if heading == 0:
                pos_x -= 1
            elif heading == 1:
                pos_y += 1
            elif heading == 2:
                pos_x += 1
            elif heading == 3:
                pos_y -= 1

            for drape in drapes:
                if drape.curtain[(pos_x, pos_y)]:  # we have a hit!
                    drape.curtain[(pos_x, pos_y)] = False
                    done = True
                    the_plot.add_reward(drape.value)
                    the_plot["num_shot"] += 1
                    if the_plot["num_shot"] >= SHOOTER_NUM_OBJECTS_PER:
                        the_plot.terminate_episode()
                    break


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
        if actions is None:  # dummy step at start of game
            return
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
        player_position = things[AGENT_CHAR].position

        if self.curtain[player_position]:
            the_plot.add_reward(self._value)
            self.curtain[player_position] = False
            the_plot["num_picked_up"] += 1
            if the_plot["num_picked_up"] == PICK_UP_NUM_OBJECTS_PER:
                the_plot.terminate_episode()


class PushableDrape(plab_things.Drape):
    def __init__(self, curtain, character, value):
        super(PushableDrape, self).__init__(curtain, character)
        self._value = value

    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_position = things[AGENT_CHAR].position

        rows, cols = player_position
        proposed_position = None
        if actions == 0:    # up 
          if self.curtain[rows - 1, cols]: 
              position = rows - 1, cols
              proposed_position = rows - 2, cols
        elif actions == 1:  # right 
          if self.curtain[rows, cols + 1]: 
              position = rows, cols + 1
              proposed_position = rows, cols + 2
        elif actions == 2:  # down 
          if self.curtain[rows + 1, cols]:
              position = rows + 1, cols
              proposed_position = rows + 2, cols
        elif actions == 3:  # left 
          if self.curtain[rows, cols - 1]: 
              position = rows, cols - 1
              proposed_position = rows, cols - 2

        if proposed_position is None:
            return 

        # check if reached edge
        proposed_row, proposed_col = proposed_position
        if proposed_row == 0 or proposed_row == GRID_SIZE + 1 or proposed_col == 0 or proposed_col == GRID_SIZE + 1:
            the_plot.add_reward(self._value)
            self.curtain[position] = False
            the_plot["num_pushed_off"] += 1
            if the_plot["num_pushed_off"] == PUSHER_NUM_OBJECTS_PER:
                the_plot.terminate_episode()
        else:
            for thing_key, thing in things.items():
                if thing_key == AGENT_CHAR:
                    continue
                if thing.curtain[proposed_position]:
                    the_plot["player_blocked"] = True
                    break
            else:
                self.curtain[position] = False
                self.curtain[proposed_position] = True
            

class ShootableDrape(plab_things.Drape):
    def __init__(self, curtain, character, value):
        super(ShootableDrape, self).__init__(curtain, character)
        self._value = value

    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_position = things[AGENT_CHAR].position

        if self.curtain[player_position]:
            the_plot.add_reward(NEG_VALUE)
            the_plot.terminate_episode()

#        if the_plot["shooting"]:
#            player_heading = the_plot["heading"]

#            the_plot.add_reward(self._value)
#            self.curtain[player_position] = False
#            the_plot["num_picked_up"] += 1
#            if the_plot["num_picked_up"] == PICK_UP_NUM_OBJECTS_PER:

    @property
    def value(self):
        return self._value


def curse_color(c):
    return tuple(int(999 * x) for x in c)


class Renderer(object):
    """Renders an observation into an RGB image"""
    def __init__(self, object_dict, scroll_size=SCROLL_SIZE,
                 upsample_size=UPSAMPLE_SIZE, agent_shape=AGENT_SHAPE,
                 scale=1.): 

        cropper = cropping.ScrollingCropper(
            rows=scroll_size, cols=scroll_size, to_track=[AGENT_CHAR],
            pad_char=BG_CHAR, scroll_margins=(None, None))
        self.cropper = cropper
        self.upsample_size = upsample_size 
        self.scroll_size = scroll_size
        self.scale = scale
        ones_square = np.ones([upsample_size, upsample_size], np.float32)
        agent_shape = self._render_plain_shape(agent_shape)
        self.decoder_dict = {
            ord(WALL_CHAR): ones_square[:, :, None] * np.array(
                COLOURS[WALL_CHAR])[None, None, :] * scale,
            ord(AGENT_CHAR): agent_shape[:, :, None] * np.array(
                COLOURS[AGENT_CHAR])[None, None, :] * scale,
        }
        self.bg_ord = ord(BG_CHAR)
        self.agent_ord = ord(AGENT_CHAR)
        for name, properties in object_dict.items():
            _, shape_name = name.split("_")
            raw_shape = self._render_plain_shape(shape_name)
            raw_color = np.array(properties["color"], np.float32)
            this_image = raw_shape[:, :, None] * raw_color[None, None, :] 
            self.decoder_dict[ord(properties["char"])] = this_image * scale

    def _render_plain_shape(self, name):
        """Shape without color dimension"""
        size = self.upsample_size
        shape = np.zeros([size, size], np.float32)
        if name == "square":
            shape[:, :] = 1.
        elif name == "diamond":
            for i in range(size):
                for j in range(size):
                    if np.abs(i - size // 2) + np.abs(j - size // 2) <= size // 2:
                        shape[i, j] = 1.
        elif name == "triangle":
            for i in range(size):
                for j in range(size):
                    if np.abs(j - size // 2) - np.abs(i // 2) < 1:
                        shape[i, j] = 1.
        elif name == "tee":
            shape[:, size // 2 - size // 6: size // 2 + size //6 + 1] = 1.
            shape[0:size//3 + 1, :] = 1.
        return shape

    def __call__(self, obs, heading=0):
        board = self.cropper.crop(obs).board
        image = np.zeros([self.scroll_size * self.upsample_size,
                          self.scroll_size * self.upsample_size,
                          3], np.float32)
        for i in range(self.scroll_size):
            for j in range(self.scroll_size):
                this_char = board[i, j]
                if this_char == self.bg_ord:
                    continue
                this_obj = self.decoder_dict[this_char]
                image[i * self.upsample_size:(i + 1) * self.upsample_size,
                      j * self.upsample_size:(j + 1) * self.upsample_size,
                      :] = this_obj
        if heading != 0:
            image = np.rot90(image, heading)
            # fix agent orientation
            middle = (self.scroll_size // 2) * self.upsample_size
            agent = image[middle:middle + self.upsample_size,
                          middle:middle + self.upsample_size,
                          :]
            image[middle:middle + self.upsample_size,
                  middle:middle + self.upsample_size,
                  :] = np.rot90(agent, 4 - heading) 
        return image
        

class GameDef(object):
    def __init__(self, game_type, good_color, bad_color, switched_colors,
                 switched_left_right):
        self.game_type = game_type
        self.good_color = good_color
        self.bad_color = bad_color
        self.switched_colors = switched_colors
        self.switched_left_right = switched_left_right

    def __str__(self):
        return "{}_{}_{}_{}_{}".format(self.game_type,
                                       self.good_color,
                                       self.bad_color,
                                       self.switched_colors,
                                       self.switched_left_right)

    def __eq__(self, other):
        if not isinstance(other, GameDef):
            return False
        return (self.game_type == other.game_type) and (self.good_color == other.good_color) and (self.bad_color == other.bad_color) and (self.switched_colors == other.switched_colors) and (self.switched_left_right == other.switched_left_right)


class Environment(object):
    """A game wrapper that handles resetting, rendering, and input flips."""
    def __init__(self,
                 game_def, 
                 num_actions=8,
                 max_steps=40,
                 objects=OBJECTS):
        self.num_actions = num_actions
        self.objects = objects
        self.max_steps = max_steps
        self.renderer = Renderer(objects)

        self.game_def = game_def
        self.game_type = game_def.game_type
        self.switched_left_right = game_def.switched_left_right
        action_dict = {i: i for i in range(num_actions)}
        if self.switched_left_right:
            action_dict[1] = 3
            action_dict[3] = 1
        self.action_map = lambda a: action_dict[a] 

    def reset(self):
        game_def = self.game_def
        game = make_game(game_def.game_type,
                         game_def.good_color,
                         game_def.bad_color,
                         game_def.switched_colors)
        self._game = game
        self.renderer.cropper.set_engine(self._game)
        self.step_count = 0

        raw_obs, _, _ = self._game.its_showtime()
        reward = 0.
        if self.game_type == "shooter":
            heading = self._game.the_plot["heading"]
        else:
            heading = 0
        obs = self.renderer(raw_obs, heading)
        return obs, reward, False

    def step(self, action):
        raw_obs, raw_reward, _ = self._game.play(self.action_map(action))
        reward = 0. if raw_reward is None else raw_reward
        if self.game_type == "shooter":
            heading = self._game.the_plot["heading"]
        else:
            heading = 0
        obs = self.renderer(raw_obs, heading)
        self.step_count += 1
        done = self._game.game_over or self.step_count > self.max_steps 
        return obs, reward, done

    def sample_action(self):
        return np.random.randint(self.num_actions)

    def __str__(self):
        return str(self.game_def)


def main(argv=()):
    np.random.seed(0)
    for game_type, cols in zip(["pusher", "pick_up"], 
                               [("red", "blue"), 
                                ("forest", "orange")]):
        env = Environment(GameDef(game_type, cols[0], cols[1], False, False))
        obs, r, done = env.reset()
        fig = plot.figure(frameon=False)
        fig.set_size_inches(3, 3)
        ax = plot.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plot.imshow(obs, aspect='auto')
        plot.savefig("%s_0.png" % game_type)
        for i in range(5):
            obs, r, done = env.step(i)
            fig = plot.figure(frameon=False)
            fig.set_size_inches(3, 3)
            ax = plot.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plot.imshow(obs, aspect='auto')
            plot.savefig("%s_%i.png" % (game_type, i + 1))

    for game_type in ["pusher", "shooter", "sequence_imitation", "pick_up"]:
        game = make_game(game_type, "red", "blue")

        ui = human_ui.CursesUi(
            keys_to_actions={curses.KEY_UP: 0, curses.KEY_RIGHT: 1,
                             curses.KEY_DOWN: 2, curses.KEY_LEFT: 3,
                             'w': 4, 'd': 5, 's': 6, 'a': 7,
                             'q': 8, 'Q': 8},
            delay=None, colour_fg={k: curse_color(v) for k, v in COLOURS.items()},
            colour_bg={AGENT_CHAR: (0, 0, 0)})

        ui.play(game)


if __name__ == '__main__':
    main(sys.argv)
