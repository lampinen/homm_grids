import numpy as np

import grid_tasks as grid_tasks


def switch_good_bad(game_def):
    return grid_tasks.GameDef(
        game_type=game_def.game_type,
        good_object_color=game_def.good_object_color,
        bad_object_color=game_def.bad_object_color,
        switched_good_bad=not game_def.switched_good_bad,
        switched_left_right=game_def.switched_left_right)
        

def switch_left_right(game_def):
    return grid_tasks.GameDef(
        game_type=game_def.game_type,
        good_object_color=game_def.good_object_color,
        bad_object_color=game_def.bad_object_color,
        switched_good_bad=game_def.switched_good_bad,
        switched_left_right=not game_def.switched_left_right)

def change_colors(game_def, colors_1_good, colors_1_bad,
                     colors_2_good, colors_2_bad):
    if game_def.good_object_color != colors_1_good or game_def.bad_object_color != colors_1_bad:
        return None
    return grid_tasks.GameDef(
        game_type=game_def.game_type,
        good_object_color=colors_2_good,
        bad_object_color=colors_2_bad,
        switched_good_bad=game_def.switched_good_bad,
        switched_left_right=game_def.switched_left_right)

def generate_meta_pairings(meta_mappings, train_environment_defs,
                           eval_environment_defs):
    meta_pairings = {}
    for meta_task in meta_mappings:
        meta_pairings[meta_task] = {"train": [],
                                    "eval": []}
        if meta_task == "switch_good_bad":
            mapping = switch_good_bad
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
