import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import grid_tasks 
from meta_tasks import generate_meta_pairings

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


class random_agent(object):
    """Base class for agents/random agent"""
    def __init__(self, name="random_agent"):
        self.name = name 
        self.epsilon = 1.
        self.memory_buffers = {}
        self.environment_indices = {}
        self.num_environments = 0

    def choose_action(self, environment, observation, cached=False,
                      from_embedding=None):
        action = environment.sample_action()
        return action

    def _environment_lookup(self, environment_def):
        if isinstance(environment_def, str):
            environment_name = environment_def
        else:
            environment_name = str(environment_def)

        if environment_name not in self.memory_buffers:
            self.memory_buffers[environment_name] = memory_buffer()
            self.environment_indices[environment_name] = self.num_environments 
            self.num_environments += 1
        
        return (str(environment_def),
                self.memory_buffers[environment_name],
                self.environment_indices[environment_name])

    def play(self, environment, max_steps=1e5, remember=True,
             cached=False, from_embedding=None, print_Qs=False):
        (environment_name,
         memory_buffer,
         env_index) = self._environment_lookup(environment)
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
            # add final observation
            memory_buffer.add((obs, 
                               None,
                               None)) 
            memory_buffer.end_experience()

        return done, step, total_return

    def train(self, environments_to_train):
        pass

    def do_eval(self, environments, num_games=1, max_steps=1e5, cached=False, print_Qs=False):
        names = []
        steps_mean = []
        returns_mean = []
        steps_se = []
        returns_se = []
        sqrt_n = np.sqrt(num_games)
        for e in environments:
            names.append(str(e))
            this_steps = []
            this_returns = []
            for _ in range(num_games):
                done, step, total_return = self.play(e, max_steps=max_steps, 
                                                     remember=False,
                                                     cached=cached,
                                                     print_Qs=print_Qs)
                this_steps.append(step)
                this_returns.append(total_return)
            steps_mean.append(np.mean(this_steps))
            steps_se.append(np.std(this_steps)/sqrt_n)
            returns_mean.append(np.mean(this_returns))
            returns_se.append(np.std(this_returns)/sqrt_n)
        return names, steps_mean, steps_se, returns_mean, returns_se


class EML_DQN_agent(random_agent):
    """Embedded Meta-Learning agent running a DQN-like algorithm."""
    def __init__(self, config, train_environments, eval_environments,
                 name="EML_DQN_agent"):
        super(EML_DQN_agent, self).__init__(
            name=name)
        self.config = config
        self.train_environments = train_environments
        self.eval_environments = eval_environments
        self.name_to_environment = {}
        for e in self.train_environments + self.eval_environments:
            (environment_name,
             memory_buffer,
             env_index) = self._environment_lookup(e)
            self.name_to_environment[environment_name] = e

        self.meta_batch_size = config["meta_batch_size"]
        self.discount = config["discount"]
        self.verbose = False
        self.softmax_policy = config["softmax_policy"]

        self.train_meta = config["train_meta"]

        if self.train_meta:
            self.meta_tasks = config["meta_tasks"]
            self.meta_pairings = generate_meta_pairings(
                self.meta_tasks,
                train_environment_defs=[e.game_def for e in self.train_environments],
                eval_environment_defs=[e.game_def for e in self.eval_environments])
            self.meta_task_indices = {mt: i + self.num_environments for (i, mt) in enumerate(self.meta_tasks)}
            self.meta_dataset_cache = {
                mt: {"tr": None,
                     "ev": None} for mt in self.meta_tasks
            }
        ##### network
        internal_nonlinearity = config["internal_nonlinearity"]

        ## vision -- we follow the DQN architecture

        self.input_ph = tf.placeholder(tf.float32, [None, 91, 91, 3])
        # for efficient forward passes, don't compute all outputs,
        self.inference_input_ph = tf.placeholder(tf.float32, [1, 91, 91, 3])

        # also provide next state, for disambiguating action permutations
        self.next_state_ph = tf.placeholder(tf.float32, [None, 91, 91, 3]) 

        # preprocessing as necessary
        processed_input = self.input_ph
        self.processed_input = processed_input
        processed_inf_input = self.inference_input_ph
    
        # vision: preprocessed vision inputs -> Z 
        def _vision(processed_input, reuse=True):
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

        with tf.variable_scope("learner"): # learning DQN
            embedded_inputs = _vision(processed_input, reuse=False)
            self.embedded_inputs = embedded_inputs
            embedded_inf_inputs = _vision(processed_inf_input)
            embedded_next_states = _vision(self.next_state_ph)
        with tf.variable_scope("target"): # target DQN
            embedded_inputs_targ = _vision(processed_input, reuse=False)
            self.embedded_inputs_targ = embedded_inputs_targ
            embedded_next_states_targ = _vision(self.next_state_ph)

        self.meta_input_indices_ph = tf.placeholder(tf.int32,
                                                    [None,])
            
        ## Outcome: (Action, Reward) -> Z
        self.action_ph = tf.placeholder(tf.int32, [None,])
        self.reward_ph = tf.placeholder(tf.float32, [None,]) 

        self.preprocessed_outcomes = tf.concat(
            [tf.one_hot(self.action_ph, depth=config["num_actions"]),
             tf.expand_dims(self.reward_ph, axis=-1),
             embedded_next_states],
            axis=-1)
        self.preprocessed_outcomes_targ = tf.concat(
            [tf.one_hot(self.action_ph, depth=config["num_actions"]),
             tf.expand_dims(self.reward_ph, axis=-1),
             embedded_next_states_targ],
            axis=-1)
        
        def _outcome_encoder(outcomes, reuse=True):
            oh = outcomes
            with tf.variable_scope("outcome_enc", reuse=reuse):
                oh = slim.fully_connected(oh, config["T_num_hidden"],
                                          activation_fn=internal_nonlinearity)
                oh = slim.fully_connected(oh, config["z_dim"],
                                          activation_fn=None)
            return oh

        
        with tf.variable_scope("learner"):
            embedded_outcomes = _outcome_encoder(self.preprocessed_outcomes,
                                                 reuse=False)
        with tf.variable_scope("target"):
            embedded_outcomes_targ = _outcome_encoder(
                self.preprocessed_outcomes_targ, reuse=False)

        self.meta_target_indices_ph = tf.placeholder(tf.int32,
                                                    [None,])
        
        ## Meta: (Input, Output) -> Z (function embedding)
        self.task_index_ph = tf.placeholder(tf.int32, [None,])
        self.guess_mask_ph = tf.placeholder(tf.bool, [None,])

        def _meta_network(embedded_inputs, embedded_targets,
                          mask=self.guess_mask_ph, reuse=True):
            num_hidden_meta = config["M_num_hidden"]
            with tf.variable_scope('meta', reuse=reuse):
                meta_input = tf.concat([embedded_inputs,
                                        embedded_targets], axis=-1)
                meta_input = tf.boolean_mask(meta_input,
                                             mask)

                mh_1 = slim.fully_connected(meta_input, num_hidden_meta,
                                            activation_fn=internal_nonlinearity)
                mh_2 = slim.fully_connected(mh_1, num_hidden_meta,
                                            activation_fn=internal_nonlinearity)
                if config["meta_max_pool"]:
                    mh_2b = tf.reduce_max(mh_2, axis=0, keep_dims=True)
                else:
                    mh_2b = tf.reduce_mean(mh_2, axis=0, keep_dims=True)
                mh_3 = slim.fully_connected(mh_2b, num_hidden_meta,
                                            activation_fn=internal_nonlinearity)

                guess_embedding = slim.fully_connected(mh_3, config["z_dim"],
                                                       activation_fn=None)
                return guess_embedding

        with tf.variable_scope("learner"):
            self.base_guess_emb = _meta_network(embedded_inputs, embedded_outcomes,
                                                reuse=False)
            self.persistent_embeddings = tf.get_variable(
                "cached_task_embeddings",
                [self.num_environments + len(self.meta_tasks),
                 config["z_dim"]],
                dtype=tf.float32)
            self.update_persistent_embeddings_ph = tf.placeholder(
                tf.float32,
                [None, config["z_dim"]])

            self.update_embeddings = tf.scatter_nd_update(
                self.persistent_embeddings,
                self.task_index_ph,
                self.update_persistent_embeddings_ph)

        with tf.variable_scope("target"):
            self.base_guess_emb_targ = _meta_network(embedded_inputs_targ, 
                                                     embedded_outcomes_targ,
                                                     reuse=False)
            self.persistent_embeddings_targ = tf.get_variable(
                "cached_task_embeddings",
                [self.num_environments + len(self.meta_tasks),
                 config["z_dim"]],
                dtype=tf.float32)

        def _get_persistent_embeddings(task_indices, target_net=False):
            if target_net:
                persistent_embs = self.persistent_embeddings_targ
            else:
                persistent_embs = self.persistent_embeddings

            return tf.nn.embedding_lookup(persistent_embs,
                                          task_indices)

        

        def _get_combined_embedding_and_match_loss(guess_embedding, task_index,
                                                   guess_weight,
                                                   target_net=False):
            cached_embedding = _get_persistent_embeddings(task_index,
                                                          target_net=target_net)
            if guess_weight == 0.:
                combined_embedding = cached_embedding
                emb_match_loss = 0.
            else:
                if guess_weight == "varied":
                    guess_weight = tf.random.uniform([], dtype=tf.float32)
                combined_embedding = guess_weight * guess_embedding + (1. - guess_weight) * cached_embedding
                # could use some thought on whether to e.g. stop gradients to the
                # guess embedding here or not
                emb_match_loss = config["emb_match_loss_weight"] * tf.nn.l2_loss(
                    guess_embedding - cached_embedding) 
            return combined_embedding, emb_match_loss 


        with tf.variable_scope("learner"):
            self.lookup_cached_embs = _get_persistent_embeddings(
                self.task_index_ph, target_net=False)

            (self.base_combined_emb,
             self.base_emb_match_loss) = _get_combined_embedding_and_match_loss(
                self.base_guess_emb, self.task_index_ph,
                config["combined_emb_guess_weight"])

            meta_input_embeddings =_get_persistent_embeddings(
                self.meta_input_indices_ph) 
            meta_target_embeddings =_get_persistent_embeddings(
                self.meta_target_indices_ph) 
            self.meta_guess_emb = _meta_network(meta_input_embeddings,
                                                meta_target_embeddings)

            (self.meta_combined_emb,
             self.meta_emb_match_loss) = _get_combined_embedding_and_match_loss(
                self.meta_guess_emb, self.task_index_ph,
                config["combined_emb_guess_weight"])

        with tf.variable_scope("target"):
            (self.base_combined_emb_targ,
             _) = _get_combined_embedding_and_match_loss(
                self.base_guess_emb_targ, self.task_index_ph,
                config["combined_emb_guess_weight"])

        self.feed_embedding_ph = tf.placeholder(tf.float32, [1, config["z_dim"]])

        ## Hyper: Z -> (f: Z -> Z)
        z_dim = config["z_dim"]
        num_hidden_hyper = config["H_num_hidden"]
        num_hidden_F = config["F_num_hidden"]
        num_task_hidden_layers = config["F_num_hidden_layers"]
        tw_range = config["task_weight_weight_mult"]/np.sqrt(
            num_hidden_F * num_hidden_hyper) # yields a very very roughly O(1) map
        task_weight_gen_init = tf.random_uniform_initializer(-tw_range,
                                                             tw_range)

        def _hyper_network(function_embedding, reuse=True):
            with tf.variable_scope('hyper', reuse=reuse):
                hyper_hidden = function_embedding
                for _ in range(config["H_num_hidden_layers"]):
                    hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                        activation_fn=internal_nonlinearity)

                hidden_weights = []
                hidden_biases = []

                task_weights = slim.fully_connected(hyper_hidden, num_hidden_F*(z_dim +(num_task_hidden_layers-1)*num_hidden_F + z_dim),
                                                    activation_fn=None,
                                                    weights_initializer=task_weight_gen_init)

                task_weights = tf.reshape(task_weights, [-1, num_hidden_F, (z_dim + (num_task_hidden_layers-1)*num_hidden_F + z_dim)])
                task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden_F + z_dim,
                                                   activation_fn=None)

                Wi = tf.transpose(task_weights[:, :, :z_dim], perm=[0, 2, 1])
                bi = task_biases[:, :num_hidden_F]
                hidden_weights.append(Wi)
                hidden_biases.append(bi)
                for i in range(1, num_task_hidden_layers):
                    Wi = tf.transpose(task_weights[:, :, z_dim+(i-1)*num_hidden_F:z_dim+i*num_hidden_F], perm=[0, 2, 1])
                    bi = task_biases[:, num_hidden_F*i:num_hidden_F*(i+1)]
                    hidden_weights.append(Wi)
                    hidden_biases.append(bi)
                Wfinal = task_weights[:, :, -z_dim:]
                bfinal = task_biases[:, -z_dim:]

                for i in range(num_task_hidden_layers):
                    hidden_weights[i] = tf.squeeze(hidden_weights[i], axis=0)
                    hidden_biases[i] = tf.squeeze(hidden_biases[i], axis=0)

                Wfinal = tf.squeeze(Wfinal, axis=0)
                bfinal = tf.squeeze(bfinal, axis=0)
                hidden_weights.append(Wfinal)
                hidden_biases.append(bfinal)
                return hidden_weights, hidden_biases

        with tf.variable_scope("learner"):
            self.base_guess_task_params = _hyper_network(self.base_combined_emb,
                                                         reuse=False)
            self.cached_emb_task_params = _hyper_network(self.lookup_cached_embs)
            self.meta_guess_task_params = _hyper_network(self.meta_combined_emb)
            self.fed_emb_task_params = _hyper_network(self.feed_embedding_ph)
        with tf.variable_scope("target"):
            self.base_guess_task_params_targ = _hyper_network(
                self.base_combined_emb_targ, reuse=False)
            self.fed_emb_task_params_targ = _hyper_network(
                self.feed_embedding_ph)

        ## task network F: Z -> Z
        def _task_network(task_params, processed_input):
            hweights, hbiases = task_params
            task_hidden = processed_input
            for i in range(num_task_hidden_layers):
                task_hidden = internal_nonlinearity(
                    tf.matmul(task_hidden, hweights[i]) + hbiases[i])

            raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

            return raw_output

        # learner network
        self.base_raw_output = _task_network(self.base_guess_task_params,
                                             embedded_inputs)
        self.meta_raw_output = _task_network(self.meta_guess_task_params,
                                             meta_input_embeddings)
        self.meta_cached_emb_raw_output = _task_network(self.cached_emb_task_params,
                                                        meta_input_embeddings)
        self.fed_emb_base_raw_output = _task_network(self.fed_emb_task_params,
                                                     embedded_inputs)

        #print("embedded inf")
        #print(embedded_inf_inputs)
        self.inference_raw_output = _task_network(self.base_guess_task_params,
                                                  embedded_inf_inputs)
        self.inference_cached_embs_raw_output = _task_network(
            self.cached_emb_task_params, embedded_inf_inputs)
        self.fed_emb_inf_raw_output = _task_network(self.fed_emb_task_params,
                                                    embedded_inf_inputs)
        # target network
        self.base_raw_output_targ = _task_network(
            self.base_guess_task_params_targ,
            embedded_inputs_targ)
        self.fed_emb_base_raw_output_targ = _task_network(
            self.fed_emb_task_params_targ,
            embedded_inputs_targ)
        
        ## output mapping O: Z -> actions
        num_actions = config["num_actions"]

        def _output_mapping(Z, reuse=True):
            with tf.variable_scope('output', reuse=reuse):
                logits = slim.fully_connected(Z, num_actions,
                                              activation_fn=None)
    
                                                   
            softmax = tf.nn.softmax(config["softmax_beta"] * logits)
            return logits, softmax

        with tf.variable_scope("learner"):
            (self.base_output_logits,
             self.base_output) = _output_mapping(self.base_raw_output, 
                                                 reuse=False)
        
            (self.inference_output_logits,
             self.inference_output) = _output_mapping(self.inference_raw_output)

            (self.inference_cached_embs_output_logits,
             self.inference_cached_embs_output) = _output_mapping(
                self.inference_cached_embs_raw_output)

            (self.fed_emb_base_output_logits,
             self.fed_emb_base_output) = _output_mapping(
                self.fed_emb_base_raw_output)
        
            (self.fed_emb_inf_output_logits,
             self.fed_emb_inf_output) = _output_mapping(self.fed_emb_inf_raw_output)
        with tf.variable_scope("target"):
            (self.base_output_logits_targ,
             self.base_output_targ) = _output_mapping(self.base_raw_output_targ, 
                                                      reuse=False)
        
            (self.fed_emb_base_output_logits_targ,
             self.fed_emb_base_output_targ) = _output_mapping(
                self.fed_emb_base_raw_output_targ)

        ## losses

        self.base_target_ph = tf.placeholder(tf.float32, [None,])

        action_taken_mask = tf.one_hot(self.action_ph, depth=num_actions,
                                       on_value=True, off_value=False,
                                       dtype=tf.bool)

        base_relevant_Qs = tf.boolean_mask(self.base_output_logits,
                                           action_taken_mask)
        self.base_loss = tf.nn.l2_loss(base_relevant_Qs - self.base_target_ph) + self.base_emb_match_loss
        self.meta_loss = tf.nn.l2_loss(self.meta_raw_output - meta_target_embeddings) + self.meta_emb_match_loss

#        fed_emb_base_relevant_Qs = tf.boolean_mask(self.fed_emb_base_output_logits,
#                                           action_taken_mask)
#        self.fed_emb_base_loss = tf.nn.l2_loss(
#            fed_emb_base_relevant_Qs - self.base_target_ph)

        ## optimizer + training
        learner_vars = [v for v in tf.trainable_variables() if "learner" in v.name]
        target_vars = [v for v in tf.trainable_variables() if "target" in v.name]

        self.lr_ph = tf.placeholder(tf.float32)
        if config["optimizer"] == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        elif config["optimizer"] == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
        else: 
            raise ValueError("Unknown optimizer: %s" % config["optimizer"])
            
        self.base_train = optimizer.minimize(self.base_loss)
        self.meta_train = optimizer.minimize(self.meta_loss)

        ## copy learner to
        target_vars = [v for v in tf.trainable_variables() if "target" in v.name]
        self.update_target_op = [v_targ.assign(v) for v_targ, v in zip(target_vars, learner_vars)]

        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        print("Session")
        self.sess = tf.Session(config=sess_config)
        print("Init")
        self.sess.run(tf.global_variables_initializer())
        print("Update target net")
        self.sess.run(self.update_target_op)
        print("Ready")

    def update_target_network(self):
        self.sess.run(self.update_target_op)

    def fill_memory_buffers(self, environments, num_data_points=500,
                            random=True):
        if random:
            curr_epsilon = self.epsilon
            self.epsilon = 1.
        for environment in environments:
            print("Filling memory buffer for environment: " + str(environment))
            steps = 0.
            while steps < num_data_points:
                _, step, _ = self.play(environment, 
                                       max_steps=num_data_points - steps)
                steps += step
        if random:
            self.epsilon = curr_epsilon

    def build_feed_dict(self, memory_buffer, env_index):
        conditioning_memories = [memory_buffer.sample(2) for _ in range(self.meta_batch_size)] 
        ns_observations = np.array([x[1][0] for x in conditioning_memories])
        c_observations = np.array([x[0][0] for x in conditioning_memories])
        c_actions = np.array([x[0][1] for x in conditioning_memories], np.int32)
        c_rewards = np.array([x[0][2] for x in conditioning_memories])
        feed_dict = {
            self.action_ph: c_actions,
            self.reward_ph: c_rewards,
            self.next_state_ph: ns_observations,
            self.input_ph: c_observations,
            self.task_index_ph: np.array([env_index], dtype=np.int32),
            self.guess_mask_ph: np.ones([len(c_actions)],  # can be overridden
                                        np.bool),
        }
        return feed_dict
            
    def refresh_meta_dataset_cache(self):
        config = self.config
        for mt in self.meta_tasks:
            train_pairings = self.meta_pairings[mt]["train"]  
            num_train = len(train_pairings)
            eval_pairings = self.meta_pairings[mt]["eval"]  
            num_eval = len(eval_pairings)

            if not self.meta_dataset_cache[mt]["tr"]:
                self.meta_dataset_cache[mt]["tr"] = {}
                self.meta_dataset_cache[mt]["ev"] = {}
                self.meta_dataset_cache[mt]["tr"]["in"] = np.zeros(
                    [num_train, ], dtype=np.int32)
                self.meta_dataset_cache[mt]["tr"]["out"] = np.zeros(
                    [num_train, ], dtype=np.int32)

                self.meta_dataset_cache[mt]["ev"]["in"] = np.zeros(
                    [num_train + num_eval], dtype=np.int32)
                self.meta_dataset_cache[mt]["ev"]["out"] = np.zeros(
                    [num_train + num_eval], dtype=np.int32)
                eval_guess_mask = np.concatenate([np.ones([num_train],
                                                          dtype=np.bool),
                                                  np.zeros([num_eval],
                                                           dtype=np.bool)])

                self.meta_dataset_cache[mt]["ev"]["gm"] = eval_guess_mask 

            for i, (e, res) in enumerate(train_pairings):
                e_index = self.environment_indices[e]
                res_index = self.environment_indices[res]
                self.meta_dataset_cache[mt]["tr"]["in"][i] = e_index
                self.meta_dataset_cache[mt]["tr"]["out"][i] = res_index

            self.meta_dataset_cache[mt]["ev"]["in"][:num_train] = self.meta_dataset_cache[mt]["tr"]["in"]
            self.meta_dataset_cache[mt]["ev"]["out"][:num_train] = self.meta_dataset_cache[mt]["tr"]["out"]

            for i, (e, res) in enumerate(eval_pairings):
                e_index = self.environment_indices[e]
                res_index = self.environment_indices[res]
                self.meta_dataset_cache[mt]["ev"]["in"][num_train + i] = e_index
                self.meta_dataset_cache[mt]["ev"]["out"][num_train + i] = res_index

    def choose_action(self, environment, observation, cached=False, from_embedding=None, print_Qs=False):
        (environment_name,
         memory_buffer,
         task_index) = self._environment_lookup(environment)
        if np.random.random() < self.epsilon: 
             return environment.sample_action()
        # TODO? Have a separate memory for only actions that achieved a reward
        # condition only on these.
        if from_embedding is not None:
            feed_dict = {
                self.inference_input_ph: np.expand_dims(observation, axis=0), 
            }
            if len(from_embedding.shape) == 1:
                from_embedding = np.expand_dims(from_embedding, axis=0)
            feed_dict[self.feed_embedding_ph] = from_embedding 
            Qs, action_probs = self.sess.run(
                [self.fed_emb_inf_output_logits, self.fed_emb_inf_output], 
                feed_dict=feed_dict)
        elif cached:
            feed_dict = {
                self.inference_input_ph: np.expand_dims(observation, axis=0), 
                self.task_index_ph: np.array([task_index], dtype=np.int32),
            }
            Qs, action_probs = self.sess.run(
                [self.inference_cached_embs_output_logits,
                 self.inference_cached_embs_output], 
                feed_dict=feed_dict)
        else: # will need to remember experiences
            feed_dict = self.build_feed_dict(memory_buffer, task_index)
            feed_dict[self.inference_input_ph] = np.expand_dims(
                observation, axis=0) 
            Qs, action_probs = self.sess.run(
                [self.inference_output_logits, self.inference_output], 
                feed_dict=feed_dict)

        if print_Qs:
            print(Qs)
        action_probs = action_probs[0]
        if self.softmax_policy:
            action = np.random.choice(len(action_probs),
                                      p=action_probs)
        else:
            action = np.argmax(action_probs)

        return action

    def train_step(self, memory_buffer, task_index, lr):
        conditioning_memories = [memory_buffer.sample(3) for _ in range(self.meta_batch_size)] 
        # first run second time step from each trace through target net, to construct targets
        c_observations = np.array([x[1][0] for x in conditioning_memories])
        ns_observations = np.array([x[2][0] for x in conditioning_memories])
        c_actions = np.array([x[1][1] for x in conditioning_memories], np.int32)
        c_rewards = np.array([x[1][2] for x in conditioning_memories])
        feed_dict = {
            self.action_ph: c_actions,
            self.reward_ph: c_rewards,
            self.input_ph: c_observations,
            self.next_state_ph: ns_observations,
            self.task_index_ph: np.array([task_index], dtype=np.int32),
            self.guess_mask_ph: np.ones([len(c_actions)], 
                                        np.bool),
        }
        next_Qs = self.sess.run(self.base_output_logits_targ,
                                feed_dict=feed_dict)

        # now run the time step before, and train
        c_observations = np.array([x[0][0] for x in conditioning_memories])
        ns_observations = np.array([x[1][0] for x in conditioning_memories])
        c_actions = np.array([x[0][1] for x in conditioning_memories], np.int32)
        c_rewards = np.array([x[0][2] for x in conditioning_memories])

        targets = self.discount * np.amax(next_Qs, axis=-1) + c_rewards
        feed_dict = {
            self.action_ph: c_actions,
            self.reward_ph: c_rewards,
            self.input_ph: c_observations,
            self.next_state_ph: ns_observations,
            self.task_index_ph: np.array([task_index], dtype=np.int32),
            self.guess_mask_ph: np.concatenate([np.ones([len(c_actions) // 2], 
                                                        np.bool),
                                                np.zeros([len(c_actions) // 2],
                                                         np.bool)]),
            self.base_target_ph: targets,
            self.lr_ph: lr,
        }
        self.sess.run(self.base_train, 
                      feed_dict=feed_dict)

    def meta_train_step(self, meta_task, meta_lr):
        meta_dataset = self.meta_dataset_cache[meta_task]["tr"]
        meta_task_index = self.meta_task_indices[meta_task]
        num_tasks = len(meta_dataset["in"])
        guess_mask = np.concatenate([np.ones([num_tasks // 2], 
                                             np.bool),
                                     np.zeros([num_tasks // 2],
                                              np.bool)]) 
        np.random.shuffle(guess_mask)
        feed_dict = {
            self.meta_input_indices_ph: meta_dataset["in"],
            self.meta_target_indices_ph: meta_dataset["out"],
            self.guess_mask_ph: guess_mask,
            self.task_index_ph: np.array([meta_task_index], dtype=np.int32),
            self.lr_ph: meta_lr,
        }
        self.sess.run(self.meta_train,
                      feed_dict=feed_dict)
        
    def train_epoch(self, environments, meta_tasks, base_lr, meta_lr):
        tasks = environments + meta_tasks
        tasks = np.random.permutation(tasks)
        for t in tasks:
            if isinstance(t, str): # meta_task
                self.meta_train_step(t, meta_lr)
            else: # base task
                (environment_name,
                 memory_buffer,
                 task_index) = self._environment_lookup(t)
                self.train_step(memory_buffer, task_index, base_lr)


    def update_eval_task_embeddings(self):
        # TODO: update for held out meta-tasks (see below)
        update_inds = []
        update_values = []
        for env in self.eval_environments:
            (environment_name,
             memory_buffer,
             task_index) = self._environment_lookup(env)
            feed_dict = self.build_feed_dict(memory_buffer, task_index) 
            task_emb = self.sess.run(self.base_guess_emb, feed_dict=feed_dict)
            update_inds.append(task_index)
            update_values.append(task_emb[0])

        self.sess.run(
            self.update_embeddings,
            feed_dict={
                self.task_index_ph: np.array(update_inds, dtype=np.int32),
                self.update_persistent_embeddings_ph: update_values
            })


    def do_meta_true_eval(self, meta_tasks, cached=False, num_games=10, max_steps=1e5):
        # TODO: Update this function (or the one called next line) to allow
        # held-out meta-tasks.
        self.update_eval_task_embeddings()
        curr_epsilon = self.epsilon
        self.epsilon = 0.
        names = []
        steps_mean = []
        returns_mean = []
        steps_se = []
        returns_se = []
        sqrt_n = np.sqrt(num_games)
        for mt in meta_tasks:
            meta_dataset = self.meta_dataset_cache[mt]["ev"]
            mt_index = self.meta_task_indices[mt]
            if cached:
                feed_dict = {
                    self.meta_input_indices_ph: meta_dataset["in"],
                    self.meta_target_indices_ph: meta_dataset["out"],
                    self.task_index_ph: np.array([mt_index], dtype=np.int32),
                    self.guess_mask_ph: meta_dataset["gm"],
                }
                outputs = self.sess.run(self.meta_raw_output,
                                        feed_dict=feed_dict)
            
            else:
                feed_dict = {
                    self.meta_input_indices_ph: meta_dataset["in"],
                    self.task_index_ph: np.array([mt_index], dtype=np.int32),
                    self.guess_mask_ph: meta_dataset["gm"],
                }
                outputs = self.sess.run(self.meta_cached_emb_raw_output,
                                        feed_dict=feed_dict)

            these_pairings = self.meta_pairings[mt]["train"] +  self.meta_pairings[mt]["eval"]  
            for i, (task, mapped) in enumerate(these_pairings):
                if meta_dataset["gm"][i]: # trained
                    names.append("%s:%s->%s" % (mt, task, mapped))
                else:
                    names.append("%s[eval]:%s->%s" % (mt, task, mapped))
                mapped_env = self.name_to_environment[mapped]
                this_steps = []
                this_returns = []
                for _ in range(num_games):
                    done, step, total_return = self.play(
                        mapped_env, max_steps=max_steps, 
                        remember=False,
                        cached=False,
                        from_embedding=outputs[i, :])
                    this_steps.append(step)
                    this_returns.append(total_return)
                steps_mean.append(np.mean(this_steps))
                steps_se.append(np.std(this_steps)/sqrt_n)
                returns_mean.append(np.mean(this_returns))
                returns_se.append(np.std(this_returns)/sqrt_n)

        self.epsilon = curr_epsilon
        return names, steps_mean, steps_se, returns_mean, returns_se

    def do_eval(self, environments, num_games=10,
                max_steps=1e5, cached=False, print_Qs=False):
        curr_epsilon = self.epsilon
        self.epsilon = 0.
        results = super(EML_DQN_agent, self).do_eval(environments, num_games,
                                                     max_steps, cached=cached,
                                                     print_Qs=print_Qs)
        self.epsilon = curr_epsilon
        return results
            
