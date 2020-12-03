#!/home/lty/SoftWare/anaconda3/envs/tf/bin/python3
# -*- coding:utf-8 -*-


import math
import numpy as np
import time
import tensorflow as tf
import os
import gym
import threading

import cv2


# ******************** PARAM ********************
# -------------------- Environment    --------------------
CHECK_ERROR_TIMES   = 5
env = gym.make("Pendulum-v0")
env.seed(0)
action_dim   = env.action_space.shape[0]
state_dim    = env.observation_space.shape[0]
action_bound = np.array([env.action_space.low[0], env.action_space.high[0]])

# -------------------- Network --------------------
LEARNING_RATE       = 1e-4                  # 学习率
BATCH_SIZE          = 32                    # 批大小

ENCODE_IMG          = 128                   # 图像编码
ENCODE_DYNAMIC      = 64


# -------------------- 智能体
MAX_STEP            = 2000                   # 单幕最大数据量
GAMMA               = 0.99                  # 衰减率
TAU                 = 0.05                  # 网络的soft参数

BUFFER_MAX          = 10000                 # 经验池最大大小
EPISODE_MAX         = 10000                 # 最大幕数

POLICY_NOISE        = 1.0                   # 噪声
NOISE_CLIP          = 2.0                   # 噪声约束

ACTION_UPDATE_DELAY = 2                     # 对actor进行延迟更新

NOTDONE             = 0
SUCCESS             = 1
FAILURE             = 2

# -------------------- DEBUG
TIME_LOG_LOSS       = 50
TIME_LOG_SAVE       = 200

# ******************** The Replay Buffer ********************
class Replay_buffer():
    def __init__(self, max_size=BUFFER_MAX):
        self.size = 0
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        """
        Add the experience.\\
        Arg:\\
            data :The experience.\\
                    [s{list} ,a{np} ,r{np}, s_{list}, d{np}]\\
                    s{list} = [img3d{np}, dynamic{np}]
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
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
                reward.append(R)
                done.append(D)
            except:
                continue
        state   = np.array(state)
        next_state  = np.array(next_state)


        return state, np.array(action), np.array(reward).reshape(-1, 1), next_state, np.array(done).reshape(-1, 1)

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
    def __init__(self, act_dim,
                img_dim, dynamic_dim, 
                action_bound, 
                gamma=GAMMA,
                tau=TAU,
                actor_update_delay=ACTION_UPDATE_DELAY,
                lr_a=1e-4,
                lr_c=1e-4,
                encode_img=ENCODE_IMG,
                encode_dynamic=ENCODE_DYNAMIC,
                load_policy_model_id=0,
                load_model=False,
                model_path=None,
                log_path=None):

        # -------------------- Param --------------------
        # The param of environment
        self.action_bound_          = action_bound

        self.img_dim_               = img_dim
        self.dynamic_dim_           = dynamic_dim

        self.img_encode_len_        = encode_img
        self.dynamic_encode_len_    = encode_dynamic
        self.n_features_            = encode_img + encode_dynamic
        self.n_actions_             = act_dim

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
        self.dynamic_input_eva_     = tf.placeholder(tf.float32, shape=[None, self.dynamic_dim_] ,name="dynamic_input_eva")
        self.dynamic_input_tar_     = tf.placeholder(tf.float32, shape=[None, self.dynamic_dim_] ,name="dynamic_input_tar")

        self.reward_input_          = tf.placeholder(tf.float32, shape=[None, 1],                 name="reward")
        self.done_input_            = tf.placeholder(tf.float32, shape=[None, 1],                 name="done")
        
        # BUild the network
        with tf.variable_scope('actor'):
            self.action_eva_    = self.buildActor(self.dynamic_input_eva_, scope='eval',   trainable=True)
            self.action_tar_    = self.buildActor(self.dynamic_input_tar_, scope='target', trainable=False)
        with tf.variable_scope('critic'):
            self.Q_eva_       = self.buildCritic(self.dynamic_input_eva_, self.action_eva_, scope='eval',   trainable=True)
            self.Q_tar_       = self.buildCritic(self.dynamic_input_tar_, self.action_tar_, scope='target', trainable=False)
        # with tf.variable_scope('critic2'):
        #     self.Q_2_eva_       = self.buildCritic(self.dynamic_input_eva_, self.action_eva_, scope='eval',   trainable=True)
        #     self.Q_2_tar_       = self.buildCritic(self.dynamic_input_tar_, self.action_tar_, scope='target', trainable=False)

        # The loss of TD-error
        # Q_tar_tmp               = tf.concat([self.Q_1_tar_, self.Q_2_tar_], 1)
        # self.Q_tar              = tf.reduce_min(self.Q_1_tar_)
        self.Q_target_          = self.reward_input_ + self.gamma_ * self.Q_tar_

        # The loss function of network
        self.c_loss_            = tf.losses.mean_squared_error(labels=self.Q_target_, predictions=self.Q_eva_)
        self.a_loss_            = -tf.reduce_mean(self.Q_eva_)

        # The param of network
        self.ae_params_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
        self.at_params_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')
        self.ce_params_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/eval')
        self.ct_params_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target')

        # The operation of network
        self.update_old_act_op_ = [olda.assign((1-self.tau_)*olda+self.tau_*p) for p,olda in zip(self.ae_params_, self.at_params_)]
        self.update_old_cri_op_ = [oldc.assign((1-self.tau_)*oldc+self.tau_*p) for p,oldc in zip(self.ce_params_, self.ct_params_)]

        self.init_a_op_         = [olda.assign(p) for p,olda in zip(self.ae_params_, self.at_params_)]
        self.init_c_op_         = [oldc.assign(p) for p,oldc in zip(self.ce_params_, self.ct_params_)]

        self.a_train_op_ = tf.train.AdamOptimizer(self.lr_a_).minimize(self.a_loss_, var_list=self.ae_params_)
        self.c_train_op_ = tf.train.AdamOptimizer(self.lr_c_).minimize(self.c_loss_, var_list=self.ce_params_)

        # Create the network
        self.sess_.run(tf.global_variables_initializer())

        # The saver and tensorboard
        self.writer_ = tf.summary.FileWriter(log_path+"/logs", self.sess_.graph)  
        self.saver_ = tf.train.Saver(max_to_keep=20)
        if load_model:
            self.restoreModel(self.model_path_+"/policy_"+str(self.policy_id_))

        # Initialize the network
        self.sess_.run([self.init_a_op_, self.init_c_op_])
    
    def buildCritic(self, dynamic_input, action_input, scope, trainable):
        with tf.variable_scope(scope):
            c_act = tf.layers.dense(inputs=action_input, units=64,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="action")
            c_obs = tf.layers.dense(inputs=dynamic_input, units=64,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="observation")
            c_f1 = tf.nn.relu(c_act + c_obs, name="act_obs")

            c_out = tf.layers.dense(inputs=c_f1, units=1, 
                                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     trainable=trainable, name="output")
        return c_out
    
    def buildActor(self, dynamic_input, scope, trainable):
        with tf.variable_scope(scope):
            a_f1 = tf.layers.dense(inputs=dynamic_input, units=64, activation=tf.nn.relu, 
                                   trainable=trainable, 
                                   name="action")

            mid_act = (self.action_bound_[0] + self.action_bound_[1])/2
            rng_act = (self.action_bound_[1] - self.action_bound_[0])/2

            a_out = tf.layers.dense(inputs=a_f1, units=self.n_actions_, activation=tf.nn.tanh,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    trainable=trainable, name="dense_2")
            a_out = mid_act + rng_act*a_out
            
            return a_out

    def chooseAction(self, state):
        action = self.sess_.run(self.action_eva_, {self.dynamic_input_eva_    : state})
        return action[0]

    def trainStep(self, train_s, train_a, train_r, train_s_):
        self.train_count_ += 1
        a_loss = 0.0
        c_loss = 0.0
        _, c_loss = self.sess_.run([self.c_train_op_, self.c_loss_],
                                    feed_dict={ self.dynamic_input_eva_     : train_s,
                                                self.action_eva_            : train_a, 
                                                self.reward_input_          : train_r,
                                                self.dynamic_input_tar_     : train_s_})
        if(self.train_count_%self.actor_update_delay == 0):
            _, a_loss = self.sess_.run([self.a_train_op_, self.a_loss_], 
                                        feed_dict={self.dynamic_input_eva_  : train_s})


            self.sess_.run(self.update_old_act_op_)
            self.sess_.run(self.update_old_cri_op_)
        return a_loss, c_loss

    def saveModel(self, model_path):
        self.saver_.save(self.sess_, model_path)

    def restoreModel(self, model_path):
        self.saver_.restore(self.sess_, model_path)


# ******************** DDPG ********************
class DDPG():
    def __init__(self, 
                action_dim,
                img_dim, dynamic_dim,  
                act_bound,
                lr=1e-4,
                load_policy_model_id = 0,
                load_model=False,
                model_path=None,
                load_data=False,
                data_path=None,
                log_path=None):

        # -------------------- 基础参数 --------------------
        # 网络和训练基础数据
        self.save_policy_model_id   = load_policy_model_id
        self.data_path              = data_path
        self.model_path             = model_path
        self.log_path               = log_path
        
        # 环境
        self.env                    = env
        self.env_ok                 = False

        # 训练数据处理
        self.buffer_size            = BUFFER_MAX
        self.batch                  = BATCH_SIZE
        self.continue_failure       = 0                         # 连续失败次数
        self.check_env_times        = CHECK_ERROR_TIMES         # 最大连续失败次数
        self.delete_num             = 0                         # 经验池需要删除数据量
        self.buffer = Replay_buffer(self.buffer_size)


        # 训练次数
        self.num_training           = 0
        self.num_episodes           = 0
        self.a_loss                 = 0.0
        self.c_loss                 = 0.0

        # 物理数据
        self.act_bound              = np.array(act_bound)
        self.observation            = [[],[]]
        self.obeservation_next      = [[],[]]
        self.action                 = None
        self.var                    = 0.1*self.act_bound[1]
        self.reward                 = np.zeros((1))
                    
        # 神经网络
        self.actor_critic = ActorCriticNet(action_dim,
                                           img_dim, dynamic_dim,  
                                           action_bound=act_bound,
                                           load_model=load_model,
                                           load_policy_model_id=load_policy_model_id,
                                           model_path=self.model_path,
                                           log_path=self.log_path)


        if load_data:
            self.buffer.loadData(self.data_path)
            print("\033[1;31;40m********** Gain "+str(self.buffer.size)+"experience **********\033[0m")


    def trainACStep(self):
        #采样数据，并进行训练
        train_s, train_a, train_r, train_s_, train_d = self.buffer.sample(BATCH_SIZE)
        # print('train_s',train_s)
        # print('s_', train_s_)
        #学习一步
        a_loss, c_loss = self.actor_critic.trainStep(train_s, train_a, train_r, train_s_)
        self.a_loss += a_loss
        self.c_loss += c_loss


    def trainPolicy(self, training_num):
        for self.num_episodes in range(training_num):
            total_reward = 0
            step = 0
            self.observation = self.env.reset()
            done = False
            state   = [[], []]
            state_  = [[], []]
            self.var = 0.1*self.act_bound[1]*np.exp(-self.num_episodes/200)
            while ~done and step < MAX_STEP:
                # 当前的状态
                state    = np.expand_dims(self.observation,0)
                # 当前动作
                action  = self.actor_critic.chooseAction(state).reshape(self.actor_critic.n_actions_)
                self.action = np.clip(np.random.normal(action, self.var), self.act_bound[0], self.act_bound[1])
                # 与环境交互动作
                self.observation_next, self.reward, done, info = self.env.step(action)

                state_ = np.expand_dims(self.observation_next,0)

                exp = [self.observation, self.action, self.reward/15.0, self.observation_next, done]
                self.buffer.push(exp)
                #推进一步
                self.observation = self.observation_next
                total_reward += self.reward
                step +=1
                if self.buffer.size> 200:
                    self.trainACStep()
                if done:
                    break
            # 完成一幕数据后，进行训练
            print("***** After %d times learning, the reward is ：%f *****"%(self.num_episodes,total_reward))
            # if self.num_episodes%TIME_LOG_LOSS  == 0:
            self.c_loss = self.c_loss/step
            self.a_loss = self.actor_critic.actor_update_delay*self.a_loss/step
            print("\033[34m***** Current loss Action is %f \t Critic is %f *****\033[0m" %(self.a_loss, self.c_loss))
            self.c_loss = 0.0
            self.a_loss = 0.0
            # if self.num_episodes%TIME_LOG_SAVE == 0:
            #     self.actor_critic.saveModel(self.model_path+"/policy_"+str(self.save_policy_model_id))
            #     self.buffer.saveData(self.data_path)
            #     self.save_policy_model_id += TIME_LOG_SAVE
            #     if self.num_episodes%1000 == 0:
                    



if __name__ == '__main__':
    agent = DDPG(action_dim=action_dim, img_dim=[20, 20, 20, 1], dynamic_dim=state_dim,
                 act_bound=action_bound,
                 lr=LEARNING_RATE,
                 load_policy_model_id=0.0,
                 load_model=False, model_path="/home/lty/uav/src/reinforcement_learning/model",
                 load_data=False, data_path="/home/lty/uav/src/reinforcement_learning/exp_data",
                 log_path="/home/lty/uav/src/reinforcement_learning")

    # train = threading.Thread(target=agent.trainPolicy, args=(10000,))
    # get_env.start()
    # train.start()
    agent.trainPolicy(2000)

