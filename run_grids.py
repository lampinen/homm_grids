from itertools import permutations

from copy import deepcopy

import numpy as np
import tensorflow as tf

from HoMM import HoMM_model
from HoMM.configs import default_run_config, default_architecture_config
import grid_tasks
import meta_tasks

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir": "grids_with_library/",

    "game_types": ["pick_up", "pusher", "shooter"],
    "color_pairs": [("red", "blue"), ("green", "purple"), ("yellow", "cyan"), ("pink", "ocean"), ("forest", "orange")], # good, bad

    "hold_outs": ["shooter_red_blue_True_False", "shooter_red_blue_True_True",
                  "pusher_red_blue_True_False", "pusher_red_blue_True_True",
                  "pick_up_red_blue_True_False", "pick_up_red_blue_True_True"], 

    "meta_mappings": ["switch_colors"]

    "softmax_beta": 8,

    "init_learning_rate": 3e-5,
    "init_meta_learning_rate": 1e-6,

    "lr_decay": 0.9,
    "meta_lr_decay": 0.95,

    "lr_decays_every": 30000,
    "min_learning_rate": 1e-8,
    "min_meta_learning_rate": 1e-8,

    "num_epochs": 1000000,
    "eval_every": 4000,
    "refresh_mem_buffs_every": 1500,

    "update_target_network_every": 10000, # how many epochs between updates to the target network

    "discount": 0.85,

    "persistent_task_reps": True,
    "combined_emb_guess_weight": "varied",
    "emb_match_loss_weight": 0.5,
})

architecture_config = default_architecture_config.default_architecture_config
architecture_config.update({
   "input_shape": [91, 91, 3],
   "output_shape": [8],  

   "outcome_shape": [8 + 1],  
   "output_masking": True,

    "IO_num_hidden": 128,
    "M_num_hidden": 1024,
    "H_num_hidden": 512,
    "z_dim": 1024,
    "F_num_hidden": 128,
    "optimizer": "RMSProp",

    "meta_batch_size": 128,
})

class grids_HoMM_agent(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        super(grids_HoMM_agent, self).__init__(
            architecture_config=architecture_config, run_config=run_config)

    def _pre_build_calls(self):
        run_config = self.run_config

        # set up the base task defs

        environment_defs = []
        for game_type in run_config["game_types"]:
            for good_color, bad_color in run_config["color_pairs"]:
                for switched_colors in [False, True]:
                    for switched_left_right in [False]:#, True]:
                        environment_defs.append(grid_tasks.GameDef(
                            game_type=game_type,
                            good_color=good_color,
                            bad_color=bad_color,
                            switched_colors=switched_colors,
                            switched_left_right=switched_left_right))

        environments = [grid_tasks.Environment(e) for e in environment_defs]

        train_environments = [e for e in environments if str(e) not in run_config["hold_outs"]]
        eval_environments = [e for e in environments if str(e) in run_config["hold_outs"]]

        self.base_train_tasks = train_environments 
        self.base_eval_tasks = eval_environments

        self.env_str_to_task = {str(t): t for t in self.base_train_tasks + self.base_eval_tasks}


        # set up the meta tasks
        self.meta_class_train_tasks = [] 
        self.meta_class_eval_tasks = [] 

        self.meta_map_train_tasks = run_config["meta_mappings"] 
        self.meta_map_eval_tasks = [] 

        # set up the meta pairings 
        self.meta_pairings = meta_tasks.generate_meta_pairings(
            self.meta_map_train_tasks,
            self.base_train_tasks,
            self.base_eval_tasks)

    def get_new_memory_buffer(self):
        """Can be overriden by child"""


    def _pre_loss_calls(self):
        self.base_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_unmasked_output)

        self.base_fed_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_fed_emb_unmasked_output)

        self.base_cached_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_cached_emb_unmasked_output)

    def fill_buffers(self, num_data_points=1024):
        """Add new "experiences" to memory buffers."""

    def play_games(self, num_turns=1, epsilon=0.):

    def build_feed_dict(self, task, lr=None, fed_embedding=None,
                        call_type="base_standard_train"):
        """Build a feed dict."""
        feed_dict = super(grids_HoMM_agent, self).build_feed_dict(
            task=task, lr=lr, fed_embedding=fed_embedding, call_type=call_type)

        base_or_meta, call_type, train_or_eval = call_type.split("_")

        if base_or_meta == "base":
            outcomes = feed_dict[self.base_target_ph]
            if call_type == "standard" or train_or_eval == "eval" or not self.architecture_config["persistent_task_reps"]: 
                feed_dict[self.base_outcome_ph] = outcomes 
            targets, target_mask = self._outcomes_to_targets(outcomes)
            feed_dict[self.base_target_ph] = targets 
            feed_dict[self.base_target_mask_ph] = target_mask

        fsdklgjfklgjfkl()

        return feed_dict

    def base_eval(self, task, train_or_eval):

    def base_embedding_eval(self, embedding, task):


## stuff
for run_i in range(run_config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

    model = grids_HoMM_agent(run_config=run_config)
    model.run_training()

    tf.reset_default_graph()
