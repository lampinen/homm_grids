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
    
    "init_epsilon": 1.,  # exploration probability
    "epsilon_decay": 0.05,  # additive decay
    "min_epsilon": 0.15,
    
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

   "separate_target_network": True,  # construct a separate network for e.g. Q-learning targets

    "IO_num_hidden": 128,
    "M_num_hidden": 1024,
    "H_num_hidden": 512,
    "z_dim": 1024,
    "F_num_hidden": 128,
    "optimizer": "RMSProp",

    "meta_batch_size": 128,
})

# architecture 
def vision(processed_input, reuse=True):
    vh = processed_input
    print(vh)
    with tf.variable_scope("vision", reuse=reuse):
        for num_filt, kernel, stride in [[64,
                                          grid_tasks.UPSAMPLE_SIZE,
                                          grid_tasks.UPSAMPLE_SIZE],
                                         [64, 4, 2],
                                         [64, 3, 1]]:
            vh = slim.conv2d(vh,
                             num_outputs=num_filt,
                             kernel_size=kernel,
                             stride=stride,
                             padding="VALID",
                             activation_fn=tf.nn.relu)
            print(vh)
        vh = slim.flatten(vh)
        print(vh)
        vision_out = slim.fully_connected(vh, config["z_dim"],
                                          activation_fn=None)
        print(vision_out)
    return vision_out


# memory buffer
class memory_buffer(object):
    """An object that holds traces, controls length, and allows samples."""
    def __init__(self, max_length=1000, drop_size=100):
       self.buffer = []
       self.length = 0
       self.max_length = max_length
       self.drop_size = drop_size

    def add(self, experience):
        self.buffer.append(experience)
        self.length += 1
        if self.length >= self.max_length:
            self.length -= self.drop_size
            self.buffer = self.buffer[self.drop_size:]

    def end_experience(self):
        self.add("EOE")

    def sample(self, trace_length=2):
        while True:
            index = np.random.randint(self.length - trace_length + 1)
            result = self.buffer[index:index + trace_length]
            for i in range(trace_length):
                if result[i] == "EOE":
                    break
            else:
                # success!
                return result



class grids_HoMM_agent(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        super(grids_HoMM_agent, self).__init__(
            architecture_config=architecture_config, run_config=run_config,
            input_processor=vision)

        self.epsilon = run_config["epsilon"]

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
        return memory_buffer() 

    def _pre_loss_calls(self):
        self.base_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_unmasked_output)

        self.base_fed_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_fed_emb_unmasked_output)

        self.base_cached_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_cached_emb_unmasked_output)

    def fill_buffers(self, num_data_points=1024, random=False):
        """Add new "experiences" to memory buffers."""
        if random:
            curr_epsilon = self.epsilon
            self.epsilon = 1.

        for task in self.base_train_tasks + self.base_eval_tasks:
            steps = 0.
            while steps < num_data_points:
                _, step, _ = self.play(task, max_steps=num_data_points - steps)
                steps += step

        if random:
            self.epsilon = curr_epsilon
        
    def other_decays(self):
        if self.epsilon > self.run_config["min_epsilon"]:
            self.epsilon -= self.run_config["epsilon_decay"]

    def play(self, environment, max_steps=1e5, remember=True,
             cached=False, from_embedding=None, print_Qs=False):
        (environment_name,
         memory_buffer,
         env_index) = self.base_task_lookup(environment)
        step = 0
        done = False
        total_return = 0.
        obs, _, _ = environment.reset()

        while (not done and step < max_steps):
            step += 1
            conditioning_obs = obs
            if from_embedding is not None:
                action = self.choose_action(environment, conditioning_obs,
                                            cached=False,
                                            from_embedding=from_embedding,
                                            print_Qs=print_Qs)
            else:
                action = self.choose_action(environment, conditioning_obs,
                                            cached=cached,
                                            print_Qs=print_Qs)
            this_reward = 0.
            obs, r, done = environment.step(action)
            this_reward += r
            total_return += this_reward

            if remember:
                memory_buffer.add((conditioning_obs,
                                   action,
                                   this_reward))

        if remember:
            memory_buffer.end_experience()

        return done, step, total_return

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
