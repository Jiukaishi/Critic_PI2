import numpy as np
import gym
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle
import copy
###DDPG for lower trust learning, example 1 DDPG###
###2020/5/10###
#####################  hyper parameters  ######################
MAX_EPISODES = 2000
MAX_EP_STEPS = 100
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
ROLL_OUTS = 20
RENDER = False
ENV_NAME = "InvertedDoublePendulum-v1"
PI2_coefficient = 30
###############################  DDPG  #################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim+s_dim+a_dim+1), dtype=np.float32)
        self.dynamic_memory = np.zeros((MEMORY_CAPACITY, s_dim+s_dim+a_dim), dtype=np.float32)
        # 1(the last dimension) for reward
        self.pointer = 0
        self.dynamic_pointer = 0
        self.dynamic_memory_len = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.generate_sample_from_outside_buffer = False
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):

            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # soft replace, combine old para with new para according to coefficient TAU
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(self.q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
    def sample_action(self, s):
        return self.sess.run(self.a, {self.S: s})
    def get_state_value(self, s, a):
        return self.sess.run(self.q, {self.S:s, self.a:a})

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        #BGD
        if self.pointer<MEMORY_CAPACITY:
            size = self.pointer
        else:
            size = MEMORY_CAPACITY
        indices = np.random.choice(size, size=size)
        # indice: which memory you want to use
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def generate_sample(self, batchsize):
        indices = np.random.choice(MEMORY_CAPACITY, size=batchsize)
        # indice: which memory you want to use
        bt = copy.deepcopy(self.memory[indices, :])
        bs = copy.deepcopy(bt[:, :s_dim])
        bsa = copy.deepcopy(bt[:, : s_dim + a_dim])
        br = copy.deepcopy(bt[:, -s_dim - 1: -s_dim])
        bs_ = copy.deepcopy(bt[:, -s_dim:])
        delta = bs_ - bs
        sa = bsa
        return  sa, delta
    def generate_sample_volumn(self, volumn, batchsize):
        indices = np.random.choice(volumn, size=batchsize)
        # indice: which memory you want to use
        bt = copy.deepcopy(self.dynamic_memory[indices, :])
        bs = copy.deepcopy(bt[:, :s_dim])
        bsa = copy.deepcopy(bt[:, : s_dim + a_dim])
        br = copy.deepcopy(bt[:, -s_dim - 1: -s_dim])
        bs_ = copy.deepcopy(bt[:, -s_dim:])
        delta = bs_ - bs
        sa = bsa
        return  sa, delta
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net_1 = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            net_2 = tf.layers.dense(net_1, units=30, use_bias=True, activation=tf.nn.relu,
                                       trainable=trainable)  # 3:    60神经元&relu
            net_3 = tf.layers.dense(net_2, units=40, use_bias=True, activation=tf.nn.relu,
                                       trainable=trainable)  # 4:    40神经元&relu
            a = tf.layers.dense(net_3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(inputs=net_1, units=60, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            net_3 = tf.layers.dense(inputs=net_2, units=100, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            net_4 = tf.layers.dense(inputs=net_3, units=60, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            return tf.layers.dense(net_4, 1, trainable=trainable)  # Q(s,a)






###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high.shape
print("action bound is", a_bound)
ddpg = DDPG(a_dim, s_dim, a_bound)
################################
var = 3  # control exploration
train_time = 100
rewards = []
for episode in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if ddpg.generate_sample_from_outside_buffer == False:
            if RENDER:
                env.render()
        # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -1, 1)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)
            ddpg.store_transition(s, a, r , s_)
            if done:
                break
##################取消buffer不满不训练###############
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
    if True:
        if var >= 0.1:
            var *= .9999  # decay the action randomness
        for i in range(train_time):
            ddpg.learn()

#############test###########
    if (episode+1) % 2 == 0:
      total_reward = 0
      testtime=3
      for i in range(testtime):
        this_reward = 0
        state = env.reset()
        for j in range(MAX_EP_STEPS):
          env.render()
          action = ddpg.choose_action(state) # direct action for test
          state, reward, done,_ = env.step(action)
          total_reward += reward
          this_reward += reward
          if done:
            break
        print("ddpg reward:", this_reward)
      ave_reward = total_reward/testtime
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
      rewards.append(ave_reward)
#############end test#############
    if (episode+1) % 20 == 0:
        try:

            plt.plot(rewards, label='ddpg')
            with open('./ddpg1023/ddpg_data', 'wb') as f:
                pickle.dump(rewards, f, pickle.HIGHEST_PROTOCOL)
            plt.xlabel('every 2 new trajectories with env', fontsize=16)
            plt.ylabel('scores', fontsize=16)
            plt.legend()
            plt.savefig('./ddpg1023/ddpg.png')
            plt.clf()
            print('plot save successed')
        except:
            print('figure save failed')
