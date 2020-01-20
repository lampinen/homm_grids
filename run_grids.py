from itertools import permutations

from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from HoMM import HoMM_model
from HoMM.configs import default_run_config, default_architecture_config
import grid_tasks
import meta_tasks

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir": "/data3/lampinen/grids_presentable/basic/",

    "run_offset": 0,
    "num_runs": 1,

    "game_types": ["pick_up", "pusher"],#, "shooter"], -- if reenabled, change num of actions
    "color_pairs": [("red", "blue"), ("green", "purple"), ("yellow", "cyan"), ("pink", "ocean"), ("forest", "orange")], # good, bad

    "hold_outs": ["pusher_forest_orange_True_False",
                  "pick_up_forest_orange_True_False",
                  "pusher_red_blue_True_False",
                  "pick_up_red_blue_True_False"], 

    "max_steps": 150,

    "meta_mappings": ["switch_colors"],

    "softmax_beta": 8,
    "softmax_policy": True,

    "init_learning_rate": 3e-5,
    "init_meta_learning_rate": 3e-5,

    "lr_decay": 0.9,
    "meta_lr_decay": 0.95,

    "lr_decays_every": 30000,
    "min_learning_rate": 3e-8,
    "min_meta_learning_rate": 3e-7,

    "num_epochs": 400000,
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

   "separate_target_network": True,  # construct a separate network for e.g. Q-learning targets

    "IO_num_hidden": 128,
    "M_num_hidden": 1024,
    "H_num_hidden": 512,
    "z_dim": 512,
    "F_num_hidden": 128,
    "optimizer": "RMSProp",

    "meta_batch_size": 64,
    "meta_holdout_size": 32,

    "task_weight_weight_mult": 100.,
    "F_weight_normalization": False,
    
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

        "init_language_learning_rate": 3e-6,
        "eval_every": 500,  # things change faster with language
        "update_target_network_every": 5000,
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

        "init_language_learning_rate": 3e-6,
        "init_language_meta_learning_rate": 3e-6,
        "eval_every": 500,  # things change faster with language
        "update_target_network_every": 5000,
    })

    architecture_config.update({
        "max_sentence_len": 5,
    })



# architecture 
def vision(processed_input, z_dim, reuse=False):
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
        vision_out = slim.fully_connected(vh, z_dim,
                                          activation_fn=None)
        print(vision_out)
    return vision_out

def outcome_processor(outcomes, IO_num_hidden, z_dim,
                      internal_nonlinearity, reuse=False):
    with tf.variable_scope('outcome_processor', reuse=reuse):
        outcome_processing_1 = slim.fully_connected(
            outcomes, IO_num_hidden, activation_fn=internal_nonlinearity)
        res = slim.fully_connected(outcome_processing_1, z_dim,
                                    activation_fn=None)
    return res

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
        self.epsilon = run_config["init_epsilon"]
        self.discount = run_config["discount"]
        self.meta_holdout_size = architecture_config["meta_holdout_size"]
        self.outcome_shape = architecture_config["outcome_shape"]
        self.softmax_policy = run_config["softmax_policy"]
        self.num_games_per_eval = run_config["num_games_per_eval"]
        self.lr_decays_every = run_config["lr_decays_every"]
        self.update_target_network_every = run_config["update_target_network_every"]
        self.max_sentence_len =  architecture_config["max_sentence_len"]

        self.best_eval_indices = None
        self.best_eval_val = -np.inf 
        super(grids_HoMM_agent, self).__init__(
            architecture_config=architecture_config, run_config=run_config,
            input_processor=lambda processed_input: vision(
                processed_input, self.architecture_config["z_dim"]),
            outcome_processor=lambda x: outcome_processor(
                x, self.architecture_config["IO_num_hidden"],
                self.architecture_config["z_dim"],
                self.architecture_config["internal_nonlinearity"]))

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

        environments = [grid_tasks.Environment(e, 
                                               num_actions=4,
                                               max_steps=run_config["max_steps"]) for e in environment_defs]

        train_environments = [e for e in environments if str(e) not in run_config["hold_outs"]]
        eval_environments = [e for e in environments if str(e) in run_config["hold_outs"]]
        
        train_environment_defs = [e for e in environment_defs if str(e) not in run_config["hold_outs"]]
        eval_environment_defs = [e for e in environment_defs if str(e) in run_config["hold_outs"]]

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
            train_environment_defs,
            eval_environment_defs)

    def get_new_memory_buffer(self):
        return memory_buffer() 

    def _pre_loss_calls(self):
        self.base_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_unmasked_output)

        self.base_fed_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_fed_emb_unmasked_output)

        self.base_cached_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_cached_emb_unmasked_output)

        if self.run_config["train_language_base"]:
            self.base_lang_output_softmax = tf.nn.softmax(
                self.run_config["softmax_beta"] * self.base_lang_unmasked_output)

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
        
    def end_epoch_calls(self, epoch):
        if epoch % self.lr_decays_every == 0 and epoch > 0:
            if self.epsilon > self.run_config["min_epsilon"]:
                self.epsilon -= self.run_config["epsilon_decay"]

        if epoch % self.update_target_network_every == 0 and epoch > 0:
            self.sess.run(self.update_target_network_op)

    def play(self, environment, max_steps=1e4, remember=True,
             cached=False, from_embedding=None, print_Qs=False, from_language=False):
        (environment_name,
         memory_buffer,
         env_index) = self.base_task_lookup(environment)
        if isinstance(environment, str):
            environment = self.env_str_to_task[environment]
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
                                            from_embedding=from_embedding)
            else:
                action = self.choose_action(environment, conditioning_obs,
                                            cached=cached, from_language=from_language)
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

    def outcome_creator(self, memories, inference_observation=None):
        try:
            actions = np.array([x[1] for x in memories], np.int32)
        except IndexError as e:
            print([x for x in memories if isinstance(x, str) or len(x) < 3])
            raise e
        rewards = np.array([x[2] for x in memories])
        outcomes = np.zeros([len(memories)] + self.outcome_shape,
                            dtype=np.float32)
        outcomes[range(len(actions)), actions] = 1.
        outcomes[:, -1] = rewards
        return outcomes

    def build_feed_dict(self, task, lr=None, fed_embedding=None, inference_observation=None,
                        call_type="base_standard_train"):
        """Build a feed dict."""
        original_call_type = call_type
        base_or_meta, call_type, train_or_eval = call_type.split("_")
        if base_or_meta == "base":
            feed_dict = {}

            if base_or_meta == "base":
                task_name, memory_buffer, task_index = self.base_task_lookup(task)
                memories = [memory_buffer.sample(2) for _ in range(self.meta_batch_size + self.meta_holdout_size)]

                if train_or_eval != "inference":
                    # first need to run inference to get targets
                    target_memories = [x[1] for x in memories] 
                    target_feed_dict = {}

                    target_feed_dict[self.task_index_ph] = [task_index]
                    target_feed_dict[self.base_outcome_ph] = self.outcome_creator(target_memories) 
                    target_feed_dict[self.base_input_ph] = np.array([x[0] for x in target_memories]) 
                    target_feed_dict[self.guess_input_mask_ph] = np.ones([len(target_memories)], np.bool) 
                    target_feed_dict[self.keep_prob_ph] = 1.
                    if call_type == "lang":
                        target_feed_dict[self.lang_keep_prob_ph] = 1.
                        target_feed_dict[self.language_input_ph] = self.task_name_to_lang_input[task_name]
                        fetch = self.base_lang_output_tn
                    else:
                        fetch = self.base_cached_emb_output_tn
                    next_Qs =  self.sess.run(fetch,
                                             feed_dict=target_feed_dict)

                    # now build actual feed dict
                    prior_memories = [x[1] for x in memories] 
                    outcomes = self.outcome_creator(prior_memories)

                if train_or_eval != "inference":
                    feed_dict[self.base_outcome_ph] = outcomes 

                    feed_dict[self.base_input_ph] = np.array([x[0] for x in prior_memories])
                    rewards = outcomes[:, -1] 
                    target_values = self.discount * np.amax(next_Qs, axis=-1) + rewards

                    output_mask = outcomes[:, :self.outcome_shape[0] - 1].astype(np.bool)
                    feed_dict[self.base_target_mask_ph] = output_mask 

                    targets = np.zeros(output_mask.shape, dtype=np.float32)
                    targets[output_mask] = target_values 

                    feed_dict[self.base_target_ph] = targets 
                    feed_dict[self.guess_input_mask_ph] = self._random_guess_mask(
                        len(outcomes))
                elif call_type == "standard":
                    prior_memories = [memory_buffer.sample(1)[0] for _ in range(self.meta_batch_size)]
                    outcomes = self.outcome_creator(prior_memories + [(inference_observation, 0, 0.)])

                    guess_mask = np.ones([len(prior_memories) + 1], np.bool) 
                    guess_mask[-1] = 0.
                    feed_dict[self.guess_input_mask_ph] = guess_mask

                    feed_dict[self.base_outcome_ph] = outcomes 

                    inputs = [x[0] for x in prior_memories] + [inference_observation]
                    feed_dict[self.base_input_ph] = np.array(inputs)
                else:
                    feed_dict[self.base_input_ph] = np.array([inference_observation])
                
                if call_type == "fed":
                    if len(fed_embedding.shape) == 1:
                        fed_embedding = np.expand_dims(fed_embedding, axis=0)
                    feed_dict[self.feed_embedding_ph] = fed_embedding
                elif call_type == "lang":
                    feed_dict[self.language_input_ph] = self.task_name_to_lang_input[task_name]

                if call_type == "cached" or self.architecture_config["persistent_task_reps"]:
                    feed_dict[self.task_index_ph] = [task_index]

                if train_or_eval == "train":
                    feed_dict[self.lr_ph] = lr
                    feed_dict[self.keep_prob_ph] = self.tkp
                    if call_type == "lang":
                        feed_dict[self.lang_keep_prob_ph] = self.lang_keep_prob
                else:
                    feed_dict[self.keep_prob_ph] = 1.
                    if call_type == "lang":
                        feed_dict[self.lang_keep_prob_ph] = 1.

        else:  # meta dicts are the same
            feed_dict = super(grids_HoMM_agent, self).build_feed_dict(
                task=task, lr=lr, fed_embedding=fed_embedding, call_type=original_call_type)


        return feed_dict

    def choose_action(self, task, observation, cached=False, from_embedding=None, from_language=False):
        if np.random.random() < self.epsilon:
             return task.sample_action()

        call_str = "base_%s_inference"
        if cached:
            call_type = "cached"
        elif from_embedding is not None:
            call_type = "fed"
        elif from_language:
            call_type = "lang"
        else:
            call_type = "standard"
        call_str = call_str % call_type

        feed_dict = self.build_feed_dict(
            task, inference_observation=observation,
            fed_embedding=from_embedding, call_type=call_str)
        if from_embedding is not None:
            action_probs = self.sess.run(
                self.base_fed_emb_output_softmax,
                feed_dict=feed_dict)
        elif cached:
            action_probs = self.sess.run(
                self.base_cached_emb_output_softmax,
                feed_dict=feed_dict)
        elif from_language:
            action_probs = self.sess.run(
                self.base_lang_output_softmax,
                feed_dict=feed_dict)
        else:
            action_probs = self.sess.run(
                self.base_output_softmax,
                feed_dict=feed_dict)

        action_probs = action_probs[-1, :]
        if self.softmax_policy:
            action = np.random.choice(len(action_probs),
                                      p=action_probs)
        else:
            action = np.argmax(action_probs)

        return action

    def base_eval(self, task, train_or_eval):
        returns = np.zeros(self.num_games_per_eval)

        for game_i in range(self.num_games_per_eval):
            _, _, total_return = self.play(task,
                                           remember=False,
                                           cached=True)
            returns[game_i] = total_return
        task_name, _, _ = self.base_task_lookup(task)
        return [task_name + "_mean_rewards"], [np.mean(returns)]

    def base_language_eval(self, task, train_or_eval):
        returns = np.zeros(self.num_games_per_eval)

        for game_i in range(self.num_games_per_eval):
            _, _, total_return = self.play(task,
                                           remember=False,
                                           cached=False,
                                           from_language=True)
            returns[game_i] = total_return
        task_name, _, _ = self.base_task_lookup(task)
        return [task_name + "_mean_rewards"], [np.mean(returns)]

    def base_embedding_eval(self, embedding, task):
        returns = np.zeros(self.num_games_per_eval)

        for game_i in range(self.num_games_per_eval):
            _, _, total_return = self.play(task,
                                           remember=False,
                                           cached=False,
                                           from_embedding=embedding)
            returns[game_i] = total_return

        return [np.mean(returns)]

    def run_eval(self, epoch, print_losses=True):
        current_epsilon = self.epsilon
        self.epsilon = 0.
        super(grids_HoMM_agent, self).run_eval(epoch=epoch,
                                               print_losses=print_losses)
        self.epsilon = current_epsilon

    def run_meta_true_eval(self):
        current_epsilon = self.epsilon
        self.epsilon = 0.
        names, losses = super(grids_HoMM_agent, self).run_meta_true_eval()
        if self.best_eval_indices is None: 
            self.best_eval_indices = np.core.defchararray.find(names, "eval") != -1
        meta_eval_mean = np.mean(np.array(losses)[self.best_eval_indices])
        if self.best_eval_val < meta_eval_mean:
            self.best_eval_val = meta_eval_mean
            self.save_parameters(self.run_config["output_dir"] + "run%i_best_eval_checkpoint" % self.run_config["this_run"])
        self.epsilon = current_epsilon
        return names, losses

    def intify_task(self, task_name):  # note: only base tasks implemented at present
        words = task_name.split("_") 
        if words[0] == "pick":
            words = ["pickup"] + words[2:]

        if len(words) < self.max_sentence_len:
            words = ["PAD"] * (self.max_sentence_len - len(words)) + words

        return [self.vocab_dict[x] for x in words]
        

## stuff
for run_i in range(run_config["run_offset"], run_config["run_offset"] + run_config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

    model = grids_HoMM_agent(run_config=run_config)
    model.run_training()

    tf.reset_default_graph()
