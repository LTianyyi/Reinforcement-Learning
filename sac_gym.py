#!/home/lty/SoftWare/anaconda3/envs/tf/bin/python3
# -*- coding:utf-8 -*-


import math
import numpy as np
import time
import tensorflow as tf
import os
import gym

# ******************** The Param ********************
# -------------------- Network --------------------
LEARNING_RATE_ACTOR     = 1e-4         # learning rate - actor
LEARNING_RATE_CRITIC    = 1e-4         # learning rate - critic
LEARNING_RATE_ALPHA     = 1e-4         # learning rate - alpha

BATCH_SIZE              = 32           # size of batch

# -------------------- Agent --------------------
MAX_STEP = 2000                         # max count of step in one episode
GAMMA = 0.99                            # discount factor
TAU = 0.01                              # soft update rate of the critic
ACTION_UPDATE_DELAY = 2                 # delay step when update actor

# -------------------- Replay Buffer --------------------
BUFFER_MAX = 50000                      # max size of replay buffer


# ******************** The Replay Buffer ********************
class ReplayBuffer():
    def __init__(self, dynamic_dim, act_dim, reward_dim,
                 gamma=GAMMA, max_size=BUFFER_MAX,
                 load_data=False, data_path=None):
        """
        The replay buffer
        Arg:\\
            dynamic_dim: The dim of state.\\
            act_dim:     The dim of action.\\
            reward_dim:  The dim of reward (fot n-step).\\
            gamma:       The discount factor.\\
            max_size:    The max size of buffer.\\
            load_data:   Whether load data from path.\\
            data_path:   The path of data.\\
        """
        self.reward_dim     = reward_dim
        self.dynamic_dim    = dynamic_dim
        self.act_dim        = act_dim
        self.gamma          = gamma

        self.max_size       = max_size
        self.ptr            = 0
        self.size           = 0

        self.storage        = []
        if load_data:
            self.loadData(data_path)

    def push(self, data):
        """
        Add the experience.\\
        Arg:\\
            data :The experience.\\
                    [s{np,(dim)} ,a{np,(dim)} ,r{list,(dim)}, s_{np,(dim)}, d{bool,(1)}]\\
        """
        if self.ptr >= self.max_size:
            self.ptr = self.ptr % self.max_size
            self.storage[int(self.ptr)] = data
            self.ptr += 1
        else:
            self.storage.append(data)
            self.size += 1

    def delete(self, del_num):
        """
        Delete some data from the end in the buffer.\\
        Arg:\\
            del_num: The number of data should be deleted.
        """
        if (self.ptr < del_num):
            self.storage[int(self.max_size - del_num + self.ptr):int(self.max_size)] = []
            self.storage[int(0):int(self.ptr)] = []
        else:
            self.storage[int(self.ptr - del_num):int(self.ptr)] = []
        self.ptr = (self.max_size + self.ptr - del_num) % self.max_size
        self.size = len(self.storage)

    def sample(self, batch_size):
        """
        Sample from the buffer.\\
        Arg:
            batch_size: The number should be sampled.\\
        Return:
            [s{np(batch_size, dim)}, a{np(batch_size, dim)}, r{np(batch_size, 1)}, s_{np(batch_size, dim)}, d{np((batch_size, 1)}]
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []
        for i in ind:
            try:
                S, A, R, S_, D = self.storage[i]
                state.append(S)
                next_state.append(S_)
                action.append(A)
                rwd = 0.0
                # TODO Add importance sample !
                for i in range(self.reward_dim):
                    rwd += pow(self.gamma, i) * R[i]
                reward.append(rwd)
                done.append(D)
            except:
                continue
        state       = np.array(state).reshape(-1, self.dynamic_dim)
        next_state  = np.array(next_state).reshape(-1, self.dynamic_dim)
        reward      = np.array(reward).reshape(-1, 1)
        done        = np.array(done).reshape(-1, 1)
        action      = np.array(action).reshape(-1, self.act_dim)

        return state, action, reward, next_state, done

    def saveData(self, path):
        """
        Save the experience as 'exp_data_np.npy'.\\
        Arg:
            path: The path (without the name) where want to save.\\
        """
        np.save(path + '/exp_data_np', np.array(self.storage))

    def loadData(self, path):
        """
        Load the experience from path/exp_data_np.npy.\\
        Arg:
            path: The path (without the name) where the data saved.\\
        """
        npy = np.load(path + '/exp_data_np.npy')
        self.storage = npy.tolist()
        self.size = len(self.storage)
        self.ptr = (self.size) % self.max_size


# ******************** The Actor And The Critic ********************
class ActorCriticNet():
    def __init__(self,
                 action_dim, dynamic_dim, action_bound,
                 reward_dim, gamma, tau, actor_update_delay,
                 epsilon=1e-6,
                 lr_act=1e-4, lr_cri=1e-4, lr_alp=1e-4,
                 load_policy_model_id=0, load_model=False, model_path=None,
                 log_path=None):
        """
        The Actor and Critic.\\
        Arg:
            action_dim:             The dim of action.\\
            dynamic_dim:            The dim of state.\\
            action_bound:           The bound of action.[0] is min and [1] is max\\
            reward_dim:             The dim of reward for n-step TD.\\
            gamma:                  The discount factor.\\
            tau:                    The soft update rate of the critic.\\
            actor_update_delay:     The delay step when update actor.\\
            epsilon:                The small constant to avoid log(0).\\
            lr_act:                 The learning rate of actor.\\
            lr_cri:                 The learning rate of critic.\\
            lr_alp:                 The learning rate of alpha.\\
            load_policy_model_id:   The policy ID should be loaded or saved.\\
            load_model:             Weather to load policy.\\
            model_path:             The path of policy (without the name).\\
            log_path:               The log file of tensorboard.\\
        """
        # -------------------- Param --------------------
        # The param of environment
        self.action_bound   = action_bound

        self.reward_dim     = reward_dim
        self.dynamic_dim    = dynamic_dim
        self.action_dim     = action_dim

        # The param of reinforcement learning agent
        self.gamma              = gamma
        self.tau                = tau
        self.actor_update_delay = actor_update_delay
        self.epsilon            = epsilon
        self.min_std_log        = -20
        self.max_std_log        = 2

        self.lr_actor           = lr_act
        self.lr_critic          = lr_cri
        self.lr_alpha           = lr_alp
        self.train_count        = 0

        # The param of saving
        self.policy_id = load_policy_model_id
        self.model_path = model_path

        # -------------------- Network --------------------
        # The session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=config)

        # The input of network
        self.dynamic_input_eva = tf.placeholder(tf.float32, shape=[None, self.dynamic_dim], name="dynamic_input_eva")
        self.dynamic_input_tar = tf.placeholder(tf.float32, shape=[None, self.dynamic_dim], name="dynamic_input_tar")
        self.reward_input      = tf.placeholder(tf.float32, shape=[None, 1], name="reward")
        self.done_input        = tf.placeholder(tf.float32, shape=[None, 1], name="done")

        self.log_alpha         = tf.Variable(initial_value=[0.0], trainable=True, name="alpha_log")
        self.target_entropy    = action_dim
        self.alpha             = tf.exp(self.log_alpha, name="alpha")

        # Build the network
        with tf.variable_scope('actor'):
            self.gauss_action_eva, self.gauss_log_pi_eva, self.determinate_action_eva \
                = self.buildActor(self.dynamic_input_eva, scope='eval', trainable=True)
            self.gauss_action_tar, self.gauss_log_pi_tar, self.determinate_action_tar \
                = self.buildActor(self.dynamic_input_tar, scope='target', trainable=False)
        with tf.variable_scope('critic1'):
            self.critic1_eva = self.buildCritic(self.dynamic_input_eva, self.gauss_action_eva, scope='eval',
                                                trainable=True)
            self.critic1_tar = self.buildCritic(self.dynamic_input_tar, self.gauss_action_tar, scope='target',
                                                trainable=False)
        with tf.variable_scope('critic2'):
            self.critic2_eva = self.buildCritic(self.dynamic_input_eva, self.gauss_action_eva, scope='eval',
                                                trainable=True)
            self.critic2_tar = self.buildCritic(self.dynamic_input_tar, self.gauss_action_tar, scope='target',
                                                trainable=False)

        self.critic_tar = tf.minimum(self.critic1_tar, self.critic2_tar) - self.alpha*self.gauss_log_pi_tar
        self.critic_eva = tf.minimum(self.critic1_eva, self.critic2_eva) - self.alpha*self.gauss_log_pi_eva

        self.Q_target   = self.reward_input + (1.0 - self.done_input)*pow(self.gamma, self.reward_dim) * self.critic_tar


        # The loss function of network
        self.critic1_loss  = tf.losses.mean_squared_error(labels=self.Q_target, predictions=self.critic1_eva)
        self.critic2_loss  = tf.losses.mean_squared_error(labels=self.Q_target, predictions=self.critic2_eva)
        self.actor_loss    = -tf.reduce_mean(self.critic_eva)

        self.alpha_loss    = -tf.reduce_mean(self.alpha*(self.gauss_log_pi_eva + self.target_entropy))

        # The param of network
        self.actor_eva_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
        self.actor_tar_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        self.critic1_eva_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic1/eval')
        self.critic1_tar_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic1/target')
        self.critic2_eva_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic2/eval')
        self.critic2_tar_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic2/target')
        self.alpha_params       = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='alpha_log')


        # The operation of network
        self.update_old_actor_op   = [olda.assign((1 - self.tau) * olda + self.tau * p) \
                                      for p, olda in zip(self.actor_eva_params,     self.actor_tar_params)]
        self.update_old_critic1_op = [oldc.assign((1 - self.tau) * oldc + self.tau * p) \
                                      for p, oldc in zip(self.critic1_eva_params,   self.critic1_tar_params)]
        self.update_old_critic2_op = [oldc.assign((1 - self.tau) * oldc + self.tau * p) \
                                      for p, oldc in zip(self.critic2_eva_params,   self.critic2_tar_params)]

        self.init_a_op     = [olda.assign(p) for p, olda in zip(self.actor_eva_params,   self.actor_tar_params)]
        self.init_c1_op    = [oldc.assign(p) for p, oldc in zip(self.critic1_eva_params, self.critic1_tar_params)]
        self.init_c2_op    = [oldc.assign(p) for p, oldc in zip(self.critic2_eva_params, self.critic2_tar_params)]

        self.actor_train_op     = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss,
                                                                                 var_list=self.actor_eva_params)
        self.critic1_train_op   = tf.train.AdamOptimizer(self.lr_critic).minimize(self.critic1_loss,
                                                                                  var_list=self.critic1_eva_params)
        self.critic2_train_op   = tf.train.AdamOptimizer(self.lr_critic).minimize(self.critic2_loss,
                                                                                  var_list=self.critic2_eva_params)
        self.alpha_train_op     = tf.train.AdamOptimizer(self.lr_alpha).minimize(self.alpha_loss,
                                                                                 var_list=self.alpha_params)

        # Create the network
        self.sess.run(tf.global_variables_initializer())

        # The saver and tensorboard
        self.writer_ = tf.summary.FileWriter(log_path + "/logs", self.sess.graph)
        self.saver_ = tf.train.Saver(max_to_keep=20)
        if load_model:
            self.restoreModel(self.model_path + "/policy_" + str(self.policy_id))

        # Initialize the network
        self.sess.run([self.init_a_op, self.init_c1_op, self.init_c2_op])

    def buildCritic(self, dynamic_input, action_input, scope, trainable):
        with tf.variable_scope(scope):
            c_act = tf.layers.dense(inputs=action_input, units=256,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="action_encode")
            c_obs = tf.layers.dense(inputs=dynamic_input, units=256,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="observation_encode")
            c_f1 = tf.nn.relu(c_act + c_obs, name="act_obs_encode")

            c_out = tf.layers.dense(inputs=c_f1, units=1,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="output")
        return c_out

    def buildActor(self, dynamic_input, scope, trainable):
        with tf.variable_scope(scope):
            a_f1 = tf.layers.dense(inputs=dynamic_input, units=256, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                   bias_initializer=tf.constant_initializer(0.1),
                                   trainable=trainable,
                                   name="action_encode")

            mid_act = (self.action_bound[0] + self.action_bound[1]) / 2
            rng_act = (self.action_bound[1] - self.action_bound[0]) / 2

            mean            = tf.layers.dense(inputs=a_f1, units=self.action_dim, activation=tf.nn.tanh,
                                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                              bias_initializer=tf.constant_initializer(0.1),
                                              trainable=trainable, name="output/mean")
            log_std         = tf.layers.dense(inputs=a_f1, units=self.action_dim, activation=tf.nn.tanh,
                                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                              bias_initializer=tf.constant_initializer(0.1),
                                              trainable=trainable, name="output/log_std")

            log_std         = tf.clip_by_value(log_std, self.min_std_log, self.max_std_log, name="output/log_std_clip")
            std             = tf.exp(log_std, name="output/std")

            normal_dis      = tf.contrib.distributions.Normal(loc=mean, scale=std)

            act_sample      = normal_dis.sample(name="output/sample")
            act_sample_tanh = tf.nn.tanh(act_sample, name="output/sample_tanh")

            gauss_act       = rng_act*act_sample_tanh + mid_act
            log_prob        = normal_dis.log_prob(act_sample) - tf.log(rng_act*(1-act_sample_tanh**2) + self.epsilon)
            log_prob        = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            deter_act       = rng_act*tf.nn.tanh(mean) + mid_act
            return gauss_act, log_prob, deter_act

    def chooseGaussAction(self, state):
        action = self.sess.run(self.gauss_action_eva, {self.dynamic_input_eva: state})
        return action[0]

    def chooseDeterminateAction(self, state):
        action = self.sess.run(self.determinate_action_eva, {self.dynamic_input_eva: state})
        return action[0]

    def trainCritic(self, train_s, train_a, train_r, train_s_, train_d):
        _, _, c1_loss, c2_loss = self.sess.run([self.critic1_train_op, self.critic2_train_op,
                                                self.critic1_loss, self.critic2_loss],
                                               feed_dict={self.dynamic_input_eva  : train_s,
                                                          self.gauss_action_eva   : train_a,
                                                          self.reward_input       : train_r,
                                                          self.dynamic_input_tar  : train_s_,
                                                          self.done_input         : train_d}
                                               )
        
        return c1_loss, c2_loss

    def trainActor(self, train_s):
        _, _, a_loss, al_loss, alpha = self.sess.run([self.actor_train_op, self.alpha_train_op,
                                                      self.actor_loss, self.alpha_loss,
                                                      self.alpha],
                                                     feed_dict={self.dynamic_input_eva: train_s}
                                                     )
        self.sess.run([self.update_old_actor_op])

        return a_loss, al_loss, alpha

    def updateCriticParam(self):
        self.sess.run([self.update_old_critic1_op, self.update_old_critic2_op])

    def trainStep(self, train_s, train_a, train_r, train_s_, train_d):
        self.train_count += 1
        alpha   = 0.0
        a_loss  = 0.0
        al_loss = 0.0
        c1_loss, c2_loss = self.trainCritic(train_s, train_a, train_r, train_s_, train_d)
        if self.train_count % self.actor_update_delay == 0:
            a_loss, al_loss, alpha = self.trainActor(train_s)
            self.updateCriticParam()

        return c1_loss, c2_loss, a_loss, al_loss, alpha


    def saveModel(self, model_path):
        self.saver_.save(self.sess, model_path)

    def restoreModel(self, model_path):
        self.saver_.restore(self.sess, model_path)


# ******************** DDPG ********************
class SAC():
    def __init__(self,
                 env,
                 action_dim, dynamic_dim, act_bound,
                 reward_dim, gamma, tau, actor_update_delay,
                 max_step=MAX_STEP,
                 lr_act=1e-4, lr_cri=1e-4, lr_alp=1e-4,
                 batch_size=BATCH_SIZE,
                 buffer_max=BUFFER_MAX,
                 load_model=False, load_policy_model_id=0, model_path=None,
                 load_data=False, data_path=None,
                 log_path=None):

        # -------------------- 基础参数 --------------------
        # 环境
        self.env = env
        self.gamma = gamma

        # 训练数据处理
        self.batch_size = batch_size

        self.buffer = ReplayBuffer(dynamic_dim=dynamic_dim, act_dim=action_dim, reward_dim=reward_dim,
                                   gamma=gamma, max_size=buffer_max,
                                   load_data=load_data, data_path=data_path)

        # 训练次数
        self.num_episodes   = 0
        self.max_step       = max_step
        self.alpha          = 0.0
        self.actor_loss     = 0.0
        self.alpha_loss     = 0.0
        self.critic1_loss   = 0.0
        self.critic2_loss   = 0.0

        # 物理数据
        self.action_bound   = np.array(act_bound)
        self.dynamic_dim    = dynamic_dim
        self.action_dim     = action_dim
        self.reward_dim     = reward_dim

        self.observation        = None
        self.observation_next   = None
        self.action             = None
        self.reward             = None

        # 神经网络
        self.actor_critic = ActorCriticNet(action_dim=action_dim, dynamic_dim=dynamic_dim, action_bound=act_bound,
                                           reward_dim=reward_dim, gamma=gamma, tau=tau,
                                           actor_update_delay=actor_update_delay,
                                           epsilon=1e-6,
                                           lr_act=lr_act, lr_cri=lr_cri, lr_alp=lr_alp,
                                           load_model=load_model,
                                           load_policy_model_id=load_policy_model_id, model_path=model_path,
                                           log_path=log_path)

        if load_data:
            try:
                self.buffer.loadData(data_path)
                print("\033[1;31;40m********** Gain " + str(self.buffer.size) + "experience **********\033[0m")
            except:
                print("\033[1;31;40m********** Cannot Gain experience **********\033[0m")

    def initRecordData(self):
        self.alpha_loss = 0.0
        self.actor_loss = 0.0
        self.critic1_loss = 0.0
        self.critic2_loss = 0.0

    def trainACStep(self):
        # 采样数据，并进行训练
        train_s, train_a, train_r, train_s_, train_d = self.buffer.sample(BATCH_SIZE)
        # 学习一步
        c1_loss, c2_loss, a_loss, al_loss, alpha = \
            self.actor_critic.trainStep(train_s, train_a, train_r, train_s_, train_d)
        self.alpha          = alpha
        self.actor_loss     += a_loss
        self.critic1_loss   += c1_loss
        self.critic2_loss   += c2_loss
        self.alpha_loss     += al_loss

    def trainPolicy(self, training_num):
        for self.num_episodes in range(training_num):
            total_reward = 0
            step = 0
            done = False
            episode = []
            reward_temp = []

            self.initRecordData()
            self.observation = self.env.reset()
            while ~done and step < self.max_step:
                # 当前的状态
                state = np.expand_dims(self.observation, 0)
                self.action = self.actor_critic.chooseGaussAction(state).reshape(self.action_dim)
                self.observation_next, self.reward, done, info = self.env.step(self.action)
                step += 1

                exp = [self.observation, self.action, [], self.observation_next, done]
                reward_temp.append(self.reward / 10)
                episode.append(exp)
                if step >= self.reward_dim:
                    index_ = step - self.reward_dim
                    episode[index_][3] = episode[step - 1][3]
                    reward = np.array(reward_temp).reshape(self.reward_dim)
                    episode[index_][2] = reward
                    self.buffer.push(episode[index_])
                    reward_temp.pop(0)

                self.observation = self.observation_next
                total_reward += self.reward
                if self.buffer.size > 200:
                    self.trainACStep()
                if done:
                    break
            # 完成一幕数据后，进行训练
            print("***** After %d times learning, the reward is ：%f *****" % (self.num_episodes, total_reward))
            # if self.num_episodes%TIME_LOG_LOSS  == 0:
            self.critic2_loss = self.critic2_loss / step
            self.critic1_loss = self.critic1_loss / step
            self.actor_loss = self.actor_critic.actor_update_delay * self.actor_loss / step
            self.alpha_loss = self.actor_critic.actor_update_delay * self.alpha_loss / step
            
            print("\033[34m***** Alpha is %6f ||Loss Action is %6f \tAlpha is %6f \t Critic is %6f and %6f*****\033[0m"%
                  (self.alpha, self.actor_loss, self.alpha_loss, self.critic1_loss, self.critic2_loss))
            if self.num_episodes%500 == 0:
                self.evaluatePolicy()

    def evaluatePolicy(self):
        for i in range(10):
            total_reward = 0
            step = 0
            self.observation = self.env.reset()
            done = False
            episode = []
            reward_temp = []
            while ~done and step < MAX_STEP:
                env.render()
                # 当前的状态
                state = np.expand_dims(self.observation, 0)
                # 当前动作
                self.action = self.actor_critic.chooseDeterminateAction(state).reshape(self.action_dim)
                # 与环境交互动作
                self.observation_next, self.reward, done, info = self.env.step(self.action)
                step += 1

                exp = [self.observation, self.action, [], self.observation_next, done]
                reward_temp.append(self.reward / 10)
                episode.append(exp)
                if step >= self.reward_dim:
                    index_ = step - self.reward_dim
                    episode[index_][3] = episode[step - 1][3]
                    reward = np.array(reward_temp).reshape(self.reward_dim)
                    episode[index_][2] = reward
                    self.buffer.push(episode[index_])
                    reward_temp.pop(0)

                self.observation = self.observation_next
                total_reward += self.reward
                if done:
                    break
        print("\033[32m***** Test reward is ：%f *****\033[0m" % (total_reward))

if __name__ == '__main__':
    # -------------------- Environment    --------------------
    CHECK_ERROR_TIMES = 5
    env = gym.make("Pendulum-v0")
    env.seed(0)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_bound = np.array([env.action_space.low[0], env.action_space.high[0]])
    reward_dim = 1
    agent = SAC(env=env,
                 action_dim=action_dim, dynamic_dim=state_dim, act_bound=action_bound,
                 reward_dim=reward_dim, gamma=GAMMA, tau=TAU, actor_update_delay=ACTION_UPDATE_DELAY,
                 load_policy_model_id=0,
                 load_model=False, model_path="/home/lty/uav/src/reinforcement_learning/model",
                 load_data=False, data_path="/home/lty/uav/src/reinforcement_learning/exp_data",
                 log_path="/home/lty/uav/src/reinforcement_learning")

    # train = threading.Thread(target=agent.trainPolicy, args=(10000,))
    # get_env.start()
    # train.start()
    agent.trainPolicy(3000)

