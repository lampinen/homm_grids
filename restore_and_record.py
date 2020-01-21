import numpy as np
import tensorflow as tf
import time

import grid_tasks

from agents import random_agent, EML_DQN_agent

config = {
    'z_dim': 512, # dimension of the latent embedding space Z
    'T_num_hidden': 128, # num hidden units in outcome (target) encoder
    'M_num_hidden': 1024, # num hidden in meta network
    'H_num_hidden': 512, # " " " hyper network
    'F_num_hidden': 128, # " " " task network that H parameterizes
    'task_weight_weight_mult': 1.,
    'F_num_hidden_layers': 3,
    'H_num_hidden_layers': 3,
    'internal_nonlinearity': tf.nn.leaky_relu,
    'meta_max_pool': True, # max or average across examples
    'F_weight_normalization': False,
    'num_actions': 4,
    'softmax_beta': 8.,
    'discount': 0.85,
    'max_steps': 150,
    'meta_batch_size': 64, # how many examples the meta-net is conditioned on
                            # for base training.
    'game_types': ['pick_up', 'pusher'],#, 'shooter'], 
    'color_pairs': [('red', 'blue'), ('green', 'purple'), ('yellow', 'cyan'), ('pink', 'ocean'), ('forest', 'orange')], # good, bad
    'hold_outs': ['shooter_red_blue_True_False', 'shooter_red_blue_True_True',
                  'pusher_red_blue_True_False', 'pusher_red_blue_True_True',
                  'pick_up_red_blue_True_False', 'pick_up_red_blue_True_True'],#, 'shooter_green_purple_True_False', 'shooter_green_purple_True_True', 'shooter_yellow_teal_True_False', 'shooter_yellow_teal_True_True'], 
    'meta_tasks': ["switch_colors"],#, "switch_left_right"],
    'num_epochs': 1000000,
    'combined_emb_guess_weight': "varied", 
    'emb_match_loss_weight': 0.2,  # weight on the loss that tries to match the
                                   # embedding guess and cache
    'play_cached': False, # if true, use a cached embedding to play 
                         # (for efficiency)
    'eval_cached': True, # use cached embedding for eval 
    'print_eval_Qs': False, # for debugging
    'softmax_policy': True, # if true, sample actions from probs, else greedy
    'optimizer': 'RMSProp',
    'init_lr': 3e-5,
    'init_meta_lr': 7e-6,
    'lr_decay': 0.9,
    'meta_lr_decay': 0.95,
    'epsilon_decrease': 0.03,
    'min_epsilon': 0.15,
    'lr_decays_every': 30000,
    'min_lr': 3e-8,
    'min_meta_lr': 3e-7,
    'play_every': 1500, # how many epochs between plays
    'eval_every': 4000, # how many epochs between evals
    'update_target_network_every': 10000, # how many epochs between updates to the target network
    'train_meta': True, # whether to train meta tasks
    'restore_dir': '/data3/lampinen/grids_presentable/basic_without_library/',
    'recordings_dir': '/data3/lampinen/grids_presentable/basic_without_library/recordings/',

    'num_runs': 1,
    'run_offset': 0,

    'num_games_to_record': 10,
}

for run_i in range(config['run_offset'], 
                   config['run_offset'] + config['num_runs']):
    config["run"] = run_i
    filename_prefix = "run_%i_" % run_i
    np.random.seed(run_i)
    tf.set_random_seed(run_i)

    environment_defs = [] 
    for game_type in config["game_types"]:
        for good_color, bad_color in config["color_pairs"]:
            for switched_colors in [False, True]:
                for switched_left_right in [False]:#, True]:
                    environment_defs.append(grid_tasks.GameDef(
                        game_type=game_type,
                        good_color=good_color,
                        bad_color=bad_color,
                        switched_colors=switched_colors,
                        switched_left_right=switched_left_right))

    config["games"] = [str(e) for e in environment_defs]
    environments = [grid_tasks.Environment(e,
                                           max_steps=config["max_steps"],
                                           num_actions=config["num_actions"]) for e in environment_defs]

    train_environments = [e for e in environments if str(e) not in config["hold_outs"]]
    eval_environments = [e for e in environments if str(e) in config["hold_outs"]]

    my_agent = EML_DQN_agent(config=config,
                             train_environments=train_environments,
                             eval_environments=eval_environments)

    my_agent.restore_parameters(config["restore_dir"] + filename_prefix + "best_checkpoint")

    my_agent.fill_memory_buffers(environments)
    my_agent.refresh_meta_dataset_cache()

    meta_tasks = config["meta_tasks"]
    (m_names, m_steps_mean, m_steps_se,
     m_returns_mean, m_returns_se) = my_agent.do_meta_true_eval(
        meta_tasks, num_games=config['num_games_to_record'], record_games=True)
    print(m_returns_mean)

    my_agent.save_recordings(config['recordings_dir'], filename_prefix)

    tf.reset_default_graph()
