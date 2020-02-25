import numpy as np

import grid_tasks as grid_tasks


def switch_colors(game_def):
    return grid_tasks.GameDef(
        game_type=game_def.game_type,
        good_color=game_def.good_color,
        bad_color=game_def.bad_color,
        switched_colors=not game_def.switched_colors,
        switched_left_right=game_def.switched_left_right)
        

def switch_left_right(game_def):
    return grid_tasks.GameDef(
        game_type=game_def.game_type,
        good_color=game_def.good_color,
        bad_color=game_def.bad_color,
        switched_colors=game_def.switched_colors,
        switched_left_right=not game_def.switched_left_right)

def change_colors(game_def, colors_1_good, colors_1_bad,
                     colors_2_good, colors_2_bad):
    if game_def.good_color != colors_1_good or game_def.bad_color != colors_1_bad:
        return None
    return grid_tasks.GameDef(
        game_type=game_def.game_type,
        good_color=colors_2_good,
        bad_color=colors_2_bad,
        switched_colors=game_def.switched_colors,
        switched_left_right=game_def.switched_left_right)

def generate_meta_pairings(meta_mappings, train_environment_defs,
                           eval_environment_defs):
    meta_pairings = {}
    for meta_task in meta_mappings:
        meta_pairings[meta_task] = {"train": [],
                                    "eval": []}
        if meta_task == "switch_colors":
            mapping = switch_colors
        elif meta_task == "switch_left_right":
            mapping = switch_left_right
        elif meta_task[:6] == "change":
            contents = meta_task.split("_")
            colors_1_good, colors_1_bad = contents[1], contents[2]
            colors_2_good, colors_2_bad = contents[4], contents[5]
            mapping = lambda e: change_colors(e, colors_1_good, colors_1_bad,
                                              colors_2_good, colors_2_bad)
        else:
            raise ValueError("Unrecognized meta task: %s" % meta_task)

        for e in train_environment_defs:
            result = mapping(e)
            if result is None:
                continue
            elif result in train_environment_defs:
                meta_pairings[meta_task]["train"].append((str(e), str(result)))
            elif result in eval_environment_defs:
                meta_pairings[meta_task]["eval"].append((str(e), str(result)))
        
    return meta_pairings
