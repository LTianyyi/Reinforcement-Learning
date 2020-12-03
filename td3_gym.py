#!/home/lty/SoftWare/anaconda3/envs/tf/bin/python3
# -*- coding:utf-8 -*-


import math
import numpy as np
import time
import tensorflow as tf
import os
import gym


# ******************** PARAM ********************


# -------------------- Network --------------------
LEARNING_RATE       = 1e-4                  # 学习率
BATCH_SIZE          = 32                    # 批大小

ENCODE_IMG          = 128                   # 图像编码
ENCODE_DYNAMIC      = 64


# -------------------- 智能体
MAX_STEP            = 2000                   # 单幕最大数据量
GAMMA               = 0.99                  # 衰减率
TAU                 = 0.05                  # 网络的soft参数

BUFFER_MAX          = 50000                 # 经验池最大大小

POLICY_NOISE        = 0.1                   # 噪声
NOISE_CLIP          = 0.1                   # 噪声约束
EXPLOR_RATE         = 0.1                   # 探索率

ACTION_UPDATE_DELAY = 2                     # 对actor进行延迟更新

NOTDONE             = 0
SUCCESS             = 1
FAILURE             = 2

# -------------------- DEBUG
TIME_LOG_LOSS       = 50
TIME_LOG_SAVE       = 200

# ******************** The Replay Buffer ********************
class Replay_buffer():
    def __init__(self, dynamic_dim, act_dim, reward_dim,
                 gamma=GAMMA, max_size=BUFFER_MAX,
                 load_data=False, data_path=None):
        self.reward_dim     = reward_dim
        self.dynamic_dim    = dynamic_dim
        self.act_dim        = act_dim
        self.gamma          = gamma
        self.size = 0
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        if load_data:
            self.loadData(data_path)


    def push(self, data):
        """
        Add the experience.\\
        Arg:\\
            data :The experience.\\
                    [s{list} ,a{np} ,r{np}, s_{list}, d{np}]\\
                    s{list} = [img3d{np}, dynamic{np}]
        """
        if self.ptr >= self.max_size:
            self.ptr = self.ptr % self.max_size
            self.storage[int(self.ptr)] = data
            self.ptr += 1
        else:
            self.storage.append(data)
            self.size += 1
    
    def delete(self, del_num):
        '''
        Delete some data from the end in the buffer.\\
        Arg:\\
            del_num: The number of data deleted.
        '''
        if(self.ptr < del_num):
            self.storage[int(self.max_size-del_num+self.ptr):int(self.max_size)] = []
            self.storage[int(0):int(self.ptr)] = []
        else:
            self.storage[int(self.ptr-del_num):int(self.ptr)] = []
        self.ptr = (self.max_size + self.ptr - del_num) % self.max_size
        self.size = len(self.storage)

    def sample(self, batch_size):
        '''
        Sample from the buffer.\\
        Arg:
            batch_size: The number should be sampled.\\
        Return:
            [s{list}, a{np}, r{np}, s_{list}, d{np}]
        '''
        ind = np.random.randint(0, self.size, size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []
        for i in ind:
            try:
                S, A, R, S_, D = self.storage[i]
                state.append(S)
                next_state.append(S_)
                action.append(A)
                rwd = 0.0
                for i in range(self.reward_dim):
                    rwd += pow(self.gamma, i)*R[i]
                reward.append(rwd)
                done.append(D)
            except:
                continue
        state       = np.array(state).reshape(-1,self.dynamic_dim)
        next_state  = np.array(next_state).reshape(-1,self.dynamic_dim)
        reward = np.array(reward).reshape(-1, 1)
        done = np.array(done).reshape(-1, 1)
        action = np.array(action).reshape(-1, self.act_dim)

        return state, action, reward, next_state, done

    def saveData(self, path):
        '''
        Save the experience as 'exp_data_np'.\\
        Arg:
            path: The path (without the name) where want to save.\\
        '''
        np.save(path+'/exp_data_np', np.array(self.storage))
    
    def loadData(self, path):
        '''
        Load the experience from path/exp_data_np.npy.\\
        Arg:
            path: The path (without the name) where the data saved.\\
        '''
        npy = np.load(path+'/exp_data_np.npy')
        self.storage = npy.tolist()
        self.size    = len(self.storage)
        self.ptr     = (self.size ) % self.max_size


# ******************** The Actor And The Critic ********************
class ActorCriticNet():
    def __init__(self,
                 action_dim, dynamic_dim, action_bound,
                 reward_dim, gamma, tau, actor_update_delay,
                 lr_a=1e-4, lr_c=1e-4,
                 load_policy_model_id=0, load_model=False, model_path=None,
                 log_path=None):

        # -------------------- Param --------------------
        # The param of environment
        self.action_bound_          = action_bound

        self.reward_dim_            = reward_dim
        self.n_features_            = dynamic_dim
        self.n_actions_             = action_dim

        # The param of reinforcement learning agent
        self.gamma_                 = gamma
        self.tau_                   = tau
        self.actor_update_delay     = actor_update_delay

        self.lr_a_                  = lr_a
        self.lr_c_                  = lr_c
        self.train_count_           = 0


        # The param of sess
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess_          = tf.Session(config=config)

        # The param of saving
        self.policy_id_     = load_policy_model_id
        self.model_path_    = model_path

        # -------------------- Network --------------------
        # The input of network
        self.dynamic_input_eva_ = tf.placeholder(tf.float32, shape=[None, self.n_features_],    name="dynamic_input_eva")
        self.dynamic_input_tar_ = tf.placeholder(tf.float32, shape=[None, self.n_features_],    name="dynamic_input_tar")
        self.reward_input_      = tf.placeholder(tf.float32, shape=[None, 1],                   name="reward")
        self.done_input_        = tf.placeholder(tf.float32, shape=[None, 1],                   name="done")
        
        # Build the network
        with tf.variable_scope('actor'):
            self.action_eva_    = self.buildActor(self.dynamic_input_eva_, scope='eval',   trainable=True)
            self.action_tar_    = self.buildActor(self.dynamic_input_tar_, scope='target', trainable=False)
        with tf.variable_scope('critic1'):
            self.critic1_eva_   = self.buildCritic(self.dynamic_input_eva_, self.action_eva_, scope='eval',   trainable=True)
            self.critic1_tar_   = self.buildCritic(self.dynamic_input_tar_, self.action_tar_, scope='target', trainable=False)
        with tf.variable_scope('critic2'):
            self.critic2_eva_   = self.buildCritic(self.dynamic_input_eva_, self.action_eva_, scope='eval',   trainable=True)
            self.critic2_tar_   = self.buildCritic(self.dynamic_input_tar_, self.action_tar_, scope='target', trainable=False)

        self.critic_tar_        = tf.minimum(self.critic1_tar_, self.critic2_tar_)
        self.Q_target_          = self.reward_input_ + pow(self.gamma_, self.reward_dim_) * self.critic_tar_
        self.Q_evalue_          = tf.minimum(self.critic1_eva_, self.critic2_eva_)

        # The loss function of network
        self.c1_loss_    = tf.losses.mean_squared_error(labels=self.Q_target_, predictions=self.critic1_eva_)
        self.c2_loss_    = tf.losses.mean_squared_error(labels=self.Q_target_, predictions=self.critic2_eva_)
        self.a_loss_    = -tf.reduce_mean(self.Q_evalue_)

        # The param of network
        self.actor_eva_params_      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
        self.actor_tar_params_      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        self.critic1_eva_params_    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic1/eval')
        self.critic1_tar_params_    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic1/target')
        self.critic2_eva_params_    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic2/eval')
        self.critic2_tar_params_    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic2/target')

        # The operation of network
        self.update_old_actor_op_   = [olda.assign((1-self.tau_)*olda+self.tau_*p) \
                                        for p,olda in zip(self.actor_eva_params_, self.actor_tar_params_)]
        self.update_old_critic1_op_ = [oldc.assign((1-self.tau_)*oldc+self.tau_*p) \
                                        for p,oldc in zip(self.critic1_eva_params_, self.critic1_tar_params_)]
        self.update_old_critic2_op_ = [oldc.assign((1-self.tau_)*oldc+self.tau_*p) \
                                        for p,oldc in zip(self.critic2_eva_params_, self.critic2_tar_params_)]

        self.init_a_op_             = [olda.assign(p) for p,olda in zip(self.actor_eva_params_, self.actor_tar_params_)]
        self.init_c1_op_            = [oldc.assign(p) for p,oldc in zip(self.critic1_eva_params_, self.critic1_tar_params_)]
        self.init_c2_op_            = [oldc.assign(p) for p,oldc in zip(self.critic2_eva_params_, self.critic2_tar_params_)]

        self.a_train_op_ = tf.train.AdamOptimizer(self.lr_a_).minimize(self.a_loss_,
                                                                       var_list=self.actor_eva_params_)
        self.c1_train_op_ = tf.train.AdamOptimizer(self.lr_c_).minimize(self.c1_loss_,
                                                                        var_list=self.critic1_eva_params_)
        self.c2_train_op_ = tf.train.AdamOptimizer(self.lr_c_).minimize(self.c2_loss_,
                                                                        var_list=self.critic2_eva_params_)

        # Create the network
        self.sess_.run(tf.global_variables_initializer())

        # The saver and tensorboard
        self.writer_ = tf.summary.FileWriter(log_path+"/logs", self.sess_.graph)  
        self.saver_ = tf.train.Saver(max_to_keep=20)
        if load_model:
            self.restoreModel(self.model_path_+"/policy_"+str(self.policy_id_))

        # Initialize the network
        self.sess_.run([self.init_a_op_, self.init_c1_op_, self.init_c2_op_])
    
    def buildCritic(self, dynamic_input, action_input, scope, trainable):
        with tf.variable_scope(scope):
            c_act = tf.layers.dense(inputs=action_input, units=64,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="action_encode")
            c_obs = tf.layers.dense(inputs=dynamic_input, units=64,
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
            a_f1 = tf.layers.dense(inputs=dynamic_input, units=64, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                   bias_initializer=tf.constant_initializer(0.1),
                                   trainable=trainable, 
                                   name="action_encode")

            mid_act = (self.action_bound_[0] + self.action_bound_[1])/2
            rng_act = (self.action_bound_[1] - self.action_bound_[0])/2

            a_out = tf.layers.dense(inputs=a_f1, units=self.n_actions_, activation=tf.nn.tanh,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="output")
            a_out = mid_act + rng_act*a_out
            
            return a_out

    def chooseAction(self, state):
        action = self.sess_.run(self.action_eva_, {self.dynamic_input_eva_    : state})
        return action[0]

    def trainStep(self, train_s, train_a, train_r, train_s_, train_d):
        self.train_count_ += 1
        a_loss = 0.0
        _, _, c1_loss, c2_loss = self.sess_.run([self.c1_train_op_, self.c2_train_op_, self.c1_loss_, self.c2_loss_],
                                                feed_dict={self.dynamic_input_eva_     : train_s,
                                                           self.action_eva_            : train_a,
                                                           self.reward_input_          : train_r,
                                                           self.dynamic_input_tar_     : train_s_,
                                                           self.done_input_            : train_d})
        if(self.train_count_%self.actor_update_delay == 0):
            _, a_loss = self.sess_.run([self.a_train_op_, self.a_loss_], 
                                       feed_dict={self.dynamic_input_eva_              : train_s})

            self.sess_.run(self.update_old_actor_op_)
            self.sess_.run(self.update_old_critic1_op_)
            self.sess_.run(self.update_old_critic1_op_)
            a_loss *= self.actor_update_delay
        return a_loss, c1_loss, c2_loss

    def saveModel(self, model_path):
        self.saver_.save(self.sess_, model_path)

    def restoreModel(self, model_path):
        self.saver_.restore(self.sess_, model_path)


# ******************** DDPG ********************
class DDPG():
    def __init__(self,
                 env,
                 action_dim, dynamic_dim, act_bound,
                 reward_dim, gamma, tau, actor_update_delay,
                 lr_a=1e-4, lr_c=1e-4,
                 batch_size=BATCH_SIZE,
                 buffer_max=BUFFER_MAX,
                 load_model=False,
                 load_policy_model_id=0,
                 model_path=None,
                 load_data=False,
                 data_path=None,
                 log_path=None):

        # -------------------- 基础参数 --------------------
        # 环境
        self.env                    = env
        self.env_ok                 = False
        self.gamma                  = gamma

        # 训练数据处理
        self.batch_size = batch_size

        self.buffer     = Replay_buffer(dynamic_dim=dynamic_dim, act_dim=action_dim, reward_dim=reward_dim,
                                        gamma=gamma, max_size=buffer_max,
                                        load_data=load_data, data_path=data_path)


        # 训练次数
        self.num_episodes           = 0
        self.a_loss                 = 0.0
        self.c1_loss                = 0.0
        self.c2_loss                = 0.0

        # 物理数据
        self.action_bound           = np.array(act_bound)
        self.dynamic_dim            = dynamic_dim
        self.action_dim             = action_dim
        self.reward_dim             = reward_dim

        self.observation            = None
        self.observation_next       = None
        self.action                 = None
        self.reward                 = None
        self.var                    = None
                    
        # 神经网络
        self.actor_critic = ActorCriticNet(action_dim=action_dim,
                                           dynamic_dim=dynamic_dim,
                                           action_bound=act_bound,
                                           reward_dim=reward_dim,
                                           gamma=gamma,
                                           tau=tau,
                                           actor_update_delay=actor_update_delay,
                                           lr_a=lr_a, lr_c=lr_c,
                                           load_model=load_model,
                                           load_policy_model_id=load_policy_model_id,
                                           model_path=model_path,
                                           log_path=log_path)

        if load_data:
            try:
                self.buffer.loadData(data_path)
                print("\033[1;31;40m********** Gain "+str(self.buffer.size)+"experience **********\033[0m")
            except:
                print("\033[1;31;40m********** Cannot Gain experience **********\033[0m")


    def trainACStep(self):
        # 采样数据，并进行训练
        train_s, train_a, train_r, train_s_, train_d = self.buffer.sample(BATCH_SIZE)
        # 学习一步
        a_loss, c1_loss, c2_loss = self.actor_critic.trainStep(train_s, train_a, train_r, train_s_, train_d)
        self.a_loss += a_loss
        self.c1_loss += c1_loss
        self.c2_loss += c2_loss

    def trainPolicy(self, training_num):
        for self.num_episodes in range(training_num):
            total_reward = 0
            step = 0
            self.observation = self.env.reset()
            done = False
            self.var = (self.action_bound[1] - self.action_bound[0])*np.exp(-self.num_episodes/200)
            episode = []
            reward_temp = []
            while ~done and step < MAX_STEP:
                # 当前的状态
                state    = np.expand_dims(self.observation, 0)
                # 当前动作
                action  = self.actor_critic.chooseAction(state).reshape(self.actor_critic.n_actions_)
                self.action = np.clip(np.random.normal(action, self.var), self.action_bound[0], self.action_bound[1])
                # 与环境交互动作
                self.observation_next, self.reward, done, info = self.env.step(action)
                step +=1

                exp = [self.observation, self.action, [], self.observation_next, done]
                reward_temp.append(self.reward / 10)
                episode.append(exp)
                if step >= self.reward_dim:
                    index_              = step - self.reward_dim
                    episode[index_][3]  = episode[step-1][3]
                    reward              = np.array(reward_temp).reshape(self.reward_dim)
                    episode[index_][2]  = reward
                    self.buffer.push(episode[index_])
                    reward_temp.pop(0)
                #推进一步
                self.observation = self.observation_next
                total_reward += self.reward
                if self.buffer.size> 200:
                    self.trainACStep()
                if done:
                    break
            # 完成一幕数据后，进行训练
            print("***** After %d times learning, the reward is ：%f *****"%(self.num_episodes,total_reward))
            # if self.num_episodes%TIME_LOG_LOSS  == 0:
            self.c1_loss = self.c1_loss/step
            self.c2_loss = self.c2_loss/step
            self.a_loss = self.actor_critic.actor_update_delay*self.a_loss/step
            print("\033[34m***** Current loss Action is %f \t Critic is %f and %f*****\033[0m" %(self.a_loss, self.c1_loss, self.c2_loss))
            self.c1_loss = 0.0
            self.c2_loss = 0.0
            self.a_loss = 0.0
                    



if __name__ == '__main__':
    # -------------------- Environment    --------------------
    CHECK_ERROR_TIMES = 5
    env = gym.make("Pendulum-v0")
    env.seed(0)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_bound = np.array([env.action_space.low[0], env.action_space.high[0]])
    reward_dim = 1
    agent = DDPG(env=env,
                 action_dim=action_dim, dynamic_dim=state_dim, act_bound=action_bound,
                 reward_dim=reward_dim, gamma=GAMMA, tau=TAU, actor_update_delay=ACTION_UPDATE_DELAY,
                 load_policy_model_id=0,
                 load_model=False, model_path="/home/lty/uav/src/reinforcement_learning/model",
                 load_data=False, data_path="/home/lty/uav/src/reinforcement_learning/exp_data",
                 log_path="/home/lty/uav/src/reinforcement_learning")

    # train = threading.Thread(target=agent.trainPolicy, args=(10000,))
    # get_env.start()
    # train.start()
    agent.trainPolicy(5000)

