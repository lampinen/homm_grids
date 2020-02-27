from itertools import permutations

import time

from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation

from HoMM.configs import default_run_config, default_architecture_config
import grid_tasks
import meta_tasks

from run_grids import grids_HoMM_agent

run_config = default_run_config.default_run_config
run_config.update({
    "run_offset": 0,
    "num_runs": 1,

    "game_types": ["pick_up", "pusher"],#, "shooter"], -- if reenabled, change num of actions
    "color_pairs": [("red", "blue"), ("green", "purple"), ("yellow", "cyan"), ("pink", "ocean"), ("forest", "orange")], # good, bad

    "hold_outs": [#"pusher_forest_orange_True_False",
                  #"pick_up_forest_orange_True_False",
                  "pusher_red_blue_True_False",
                  "pick_up_red_blue_True_False"],

    "max_steps": 150,

    "meta_mappings": ["switch_colors"],

    "softmax_beta": 8,
    "softmax_policy": True,

    "init_learning_rate": 1e-4,
    "init_meta_learning_rate": 1e-4,

    "lr_decay": 0.8,
    "meta_lr_decay": 0.95,

    "lr_decays_every": 20000,
    "min_learning_rate": 1e-8,
    "min_meta_learning_rate": 1e-7,

    "num_epochs": 1000000,
    "eval_every": 4000,
    "num_games_per_eval": 10,
    "refresh_mem_buffs_every": 1500,

    "update_target_network_every": 10000, # how many epochs between updates to the target network

    "discount": 0.85,

    "init_epsilon": 1.,  # exploration probability
    "epsilon_decay": 0.03,  # additive decay
    "min_epsilon": 0.15,
})

architecture_config = default_architecture_config.default_architecture_config
architecture_config.update({
   "input_shape": [91, 91, 3],
   "output_shape": [4],  

   "outcome_shape": [4 + 1],  
   "output_masking": True,

   "mlp_output": False,

   "separate_target_network": True,  # construct a separate network for e.g. Q-learning targets

    "IO_num_hidden": 128,
    "M_num_hidden": 1024,
    "H_num_hidden": 512,
    "z_dim": 512,
    "F_num_hidden": 128,
    "F_num_hidden_layers": 3,
    "optimizer": "RMSProp",

    "meta_batch_size": 32,
    "meta_sample_size": 64,

    "task_weight_weight_mult": 10.,
    "F_weight_normalization": True,
    
    "persistent_task_reps": True,
    "combined_emb_guess_weight": "varied",
    "emb_match_loss_weight": 0.2,
})

if False:  # enable for language baseline
    run_config.update({
        "output_dir": run_config["output_dir"] + "language/",

        "train_language_base": True,
        "train_base": False,
        "train_meta": False,

        "vocab": ["pickup", "pusher"] + ["True", "False"] + list(grid_tasks.BASE_COLOURS.keys()),
        "persistent_task_reps": False,

        "init_language_learning_rate": 3e-5,
        #"eval_every": 500,  # things change faster with language
        #"update_target_network_every": 5000,
    })

    architecture_config.update({
        "max_sentence_len": 5,
    })

if False:  # enable for language base + meta 
    run_config.update({
        "output_dir": run_config["output_dir"] + "language_HoMM/",

        "train_language_base": True,
        "train_language_meta": True,
        "train_base": False,
        "train_meta": False,

        "vocab": ["PAD"] + ["switch", "colors"] + ["pickup", "pusher"] + ["True", "False"] + list(grid_tasks.BASE_COLOURS.keys()),
        "persistent_task_reps": False,

        "init_language_learning_rate": 3e-5,
        "init_language_meta_learning_rate": 3e-5,
        #"eval_every": 500,  # things change faster with language
        #"update_target_network_every": 5000,
    })

    architecture_config.update({
        "max_sentence_len": 5,
    })

recording_config = {
    "restore_dir": "/mnt/fs4/lampinen/grids_final/lessplit_wn_one_holdout/",

    "recordings_dir": "/mnt/fs4/lampinen/grids_final/lessplit_wn_one_holdout/recordings/",

    "num_runs": 5,
    "run_offset": 0,

    "num_games_to_record": 5,
}

class grids_HoMM_recording_agent(grids_HoMM_agent):
    def __init__(self, run_config, architecture_config):
        super(grids_HoMM_recording_agent, self).__init__(
            run_config=run_config,
            architecture_config=architecture_config)
        self.recordings = {}

    def play(self, environment, max_steps=1e4, remember=True,
             cached=False, from_embedding=None, print_Qs=False,
             from_language=False, record=False):
        (environment_name,
         memory_buffer,
         env_index) = self.base_task_lookup(environment)
        if isinstance(environment, str):
            environment = self.env_str_to_task[environment]
        step = 0
        done = False
        total_return = 0.
        obs, _, _ = environment.reset()
        if record:
            this_recording = [obs]

        while (not done and step < max_steps):
            step += 1
            conditioning_obs = obs
            if from_embedding is not None:
                action = self.choose_action(environment, conditioning_obs,
                                            cached=False,
                                            from_embedding=from_embedding)
            else:
                action = self.choose_action(environment, conditioning_obs,
                                            cached=cached, from_language=from_language)
            this_reward = 0.
            obs, r, done = environment.step(action)
            this_reward += r
            total_return += this_reward
            if record:
                this_recording.append(obs)

            if remember:
                memory_buffer.add((conditioning_obs,
                                   action,
                                   this_reward))

        if remember:
            memory_buffer.end_experience()

        if record:
            if environment_name not in self.recordings:
                self.recordings[environment_name] = [this_recording]
            else:
                self.recordings[environment_name].append(this_recording) 

        return done, step, total_return

    def base_embedding_eval(self, embedding, task):
        returns = np.zeros(self.num_games_per_eval)

        for game_i in range(self.num_games_per_eval):
            _, _, total_return = self.play(task,
                                           remember=False,
                                           cached=False,
                                           from_embedding=embedding,
                                           record=True)
            returns[game_i] = total_return

        return [np.mean(returns)]

    def base_language_eval(self, task, train_or_eval):
        returns = np.zeros(self.num_games_per_eval)

        for game_i in range(self.num_games_per_eval):
            _, _, total_return = self.play(task,
                                           remember=False,
                                           cached=False,
                                           from_language=True,
                                           record=True)
            returns[game_i] = total_return
        task_name, _, _ = self.base_task_lookup(task)
        return [task_name + "_mean_rewards"], [np.mean(returns)]

    def _save_recording(self, recording, filename):
        fig = plt.figure()
        p = plt.imshow(recording[0])

        def init():
            p.set_array(recording[0])
            return p,

        def frame(i):
            p.set_array(recording[i])
            return p,

        anim = matplotlib.animation.FuncAnimation(
            fig, frame, init_func=init, frames=len(recording))
        anim.save(filename, writer='imagemagick', fps=10)
        plt.close()

    def save_recordings(self, recording_path, filename_prefix):
        for env_name, recordings in self.recordings.items():
            for i, recording in enumerate(recordings):
                self._save_recording(
                    recording,
                    recording_path + filename_prefix + "%s_recording_%i.gif" % (env_name, i))



for run_i in range(recording_config["run_offset"], 
                   recording_config["run_offset"] + recording_config["num_runs"]):
    recording_config["run"] = run_i
    filename_prefix = "run%i_" % run_i
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

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

    recording_config["games"] = [str(e) for e in environment_defs]
    environments = [grid_tasks.Environment(e,
                                           max_steps=run_config["max_steps"],
                                           num_actions=architecture_config["output_shape"][0]) for e in environment_defs]

    my_agent = grids_HoMM_recording_agent(run_config=run_config, architecture_config=architecture_config)

    my_agent.restore_parameters(recording_config["restore_dir"] + filename_prefix + "best_eval_checkpoint")

    my_agent.fill_buffers()

    if run_config["train_language_base"]:
#        my_agent.update_base_task_embeddings()
#        if run_config["train_language_meta"]:
#            my_agent.update_meta_task_embeddings()
        raise NotImplementedError("Recordings for language models aren't implemented (yet)")

    my_agent.num_games_per_eval = recording_config["num_games_to_record"]
    (m_names, m_losses) = my_agent.run_meta_true_eval()

    print(m_names, m_losses)

    my_agent.save_recordings(recording_config["recordings_dir"], filename_prefix)

    tf.reset_default_graph()
