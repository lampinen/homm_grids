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
    'F_weight_normalization': False,

    'H_num_hidden_layers': 3,
    'internal_nonlinearity': tf.nn.leaky_relu,
    'meta_max_pool': True, # max or average across examples
    'num_actions': 4,
    'softmax_beta': 8.,
    'discount': 0.85,
    'max_steps': 150,
    'meta_batch_size': 64, # how many examples the meta-net is conditioned on
                            # for base training.
    'game_types': ['pick_up', 'pusher'],#, 'shooter'], 
    'color_pairs': [('red', 'blue'), ('green', 'purple'), ('yellow', 'cyan'), ('pink', 'ocean'), ('forest', 'orange')], # good, bad
    'hold_outs': ['pusher_forest_orange_True_False', 
                  'pick_up_forest_orange_True_False',
                  'pusher_red_blue_True_False', 
                  'pick_up_red_blue_True_False'],
    'meta_tasks': ["switch_colors"],#, "switch_left_right"],
    'num_epochs': 400000,
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
    'init_meta_lr': 3e-5,
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
    'results_dir': '/data3/lampinen/grids_presentable/fewer/',

    'num_runs': 5,
    'run_offset': 0,
}
#config['meta_tasks'] += ["change_%s_%s_to_%s_%s" %(cs1[0], cs1[1], cs2[0], cs2[1]) for cs1 in config['color_pairs'] for cs2 in config['color_pairs'] if cs1 != cs2]

def _save_config(filename, config):
    with open(filename, "w") as fout:
        fout.write("key, value\n")
        for key, value in config.items():
            fout.write(key + ", " + str(value) + "\n")

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

    my_agent.fill_memory_buffers(environments)
    if config["train_meta"]:
        my_agent.refresh_meta_dataset_cache()

    _save_config(config['results_dir'] + filename_prefix + 'config.csv', config)

    current_lr = config["init_lr"]
    current_meta_lr = config["init_meta_lr"]
    meta_tasks = config["meta_tasks"]
    train_meta = config["train_meta"]
    print('Running %s' % config['results_dir'])
    with open(config['results_dir'] + filename_prefix + 'base_losses.csv', 'w', buffering=1) as fout, open(config['results_dir']  + filename_prefix+ 'meta_true_losses.csv', 'w', buffering=1) as fout_meta:
        epoch = 0
        print("Evaluating...")
        (names, steps_mean, steps_se,
         returns_mean, returns_se) = my_agent.do_eval(environments, 
                                                      cached=config["eval_cached"],
                                                      print_Qs=config["print_eval_Qs"])
        names_types = ", ".join(
            [x + res_type for res_type in ["_steps_mean", "_steps_se", "_returns_mean", "_returns_se"] for x in names]) 
        print(names_types)
        results = [epoch] + steps_mean + steps_se + returns_mean + returns_se
        results_format = ', '.join(['%i'] + ['%f'] * (len(names) * 4)) + '\n'
        fout.write("epoch, " + names_types + "\n")
        print(results_format)
        print(results)
        fout.write(results_format % tuple(results))
        if train_meta:
            (m_names, m_steps_mean, m_steps_se,
             m_returns_mean, m_returns_se) = my_agent.do_meta_true_eval(meta_tasks)
            
            meta_names_types = ", ".join(
                [x + res_type for res_type in ["_steps_mean", "_steps_se", "_returns_mean", "_returns_se"] for x in m_names]) 
            print(meta_names_types)
            meta_results = [epoch] + m_steps_mean + m_steps_se + m_returns_mean + m_returns_se
            meta_results_format = ', '.join(['%i'] + ['%f'] * (len(m_names) * 4)) + '\n'
            fout_meta.write("epoch, " + meta_names_types + "\n")
            print(meta_results)

            # set up for saving "best" checkpoints
            eval_indices = np.core.defchararray.find(m_names, "[eval]") != -1
            meta_eval_mean = np.mean(np.array(m_returns_mean)[eval_indices])
            best_eval_mean = meta_eval_mean

            fout_meta.write(meta_results_format % tuple(meta_results))
        for epoch in range(1, config["num_epochs"] + 1):
            my_agent.train_epoch(train_environments,  
                                 meta_tasks if config["train_meta"] else [],
                                 current_lr,
                                 current_meta_lr)

            if epoch % config['update_target_network_every'] == 0:
                my_agent.update_target_network()

            if epoch % config['play_every'] == 0:
                print("Epoch {}".format(epoch))
                print("Playing...")
                ti = time.time()
                steps = []
                returns = []
                for e in train_environments:  # keep experience random on held-out environments 
                    _, step, tot_return = my_agent.play(
                        e, cached=config["play_cached"])
                    steps.append(step)
                    returns.append(tot_return)
                print("Play took {} seconds".format(time.time() - ti)) 

                print("Steps:")
                print(steps)
                print("Returns:")
                print(returns)

            if epoch % config['eval_every'] == 0:
                print("Evaluating...")
                ti = time.time()
                (_, steps_mean, steps_se,
                 returns_mean, returns_se) = my_agent.do_eval(
                    environments, cached=config["eval_cached"], print_Qs=config["print_eval_Qs"])
                results = [epoch] + steps_mean + steps_se + returns_mean + returns_se
                if train_meta:
                    (_, m_steps_mean, m_steps_se,
                     m_returns_mean, m_returns_se) = my_agent.do_meta_true_eval(meta_tasks)
                    meta_results = [epoch] + m_steps_mean + m_steps_se + m_returns_mean + m_returns_se
                    fout_meta.write(meta_results_format % tuple(meta_results))
                    print(meta_results)
                    meta_eval_mean = np.mean(np.array(m_returns_mean)[eval_indices])
                    if meta_eval_mean > best_eval_mean:
                        best_eval_mean = meta_eval_mean
                        my_agent.save_parameters(config['results_dir'] + filename_prefix + "best_checkpoint")
                print("Eval took {} seconds".format(time.time() - ti)) 
                fout.write(results_format % tuple(results))
                print(results)

                my_agent.save_parameters(config['results_dir'] + "latest_checkpoint")

            if epoch % config["lr_decays_every"] == 0:
                if current_lr > config["min_lr"]: 
                    current_lr *= config["lr_decay"]
                if current_meta_lr > config["min_meta_lr"]: 
                    current_meta_lr *= config["meta_lr_decay"]
                if my_agent.epsilon > config["min_epsilon"]:
                    my_agent.epsilon -= config["epsilon_decrease"]

    tf.reset_default_graph()
