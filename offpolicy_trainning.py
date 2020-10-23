import numpy as np
import gym
import tensorflow as tf
import time
import copy
from tools.env_copy import copy_env
import matplotlib.pyplot as plt
from dynamic_model import Dynamic_Net
###2020/07/08###
#####TODO: This problem is only used for offpolicy learning, once loaded the database file, it will not store episodes that created by its own.
#####################  hyper parameters  ######################
from PI2_replaybuffer import Replay_buffer

TRAIN_FROM_SCRATCH = False # 是否加载模型
MAX_EP_STEPS = 100  # 每条采样轨迹的最大长度
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99
# s_dim = env.observation_space.shape[0]
# a_dim = env.action_space.shape[0]
# a_bound = env.action_space.high.shape
VALUE_TRAIN_TIME = 50
ACTOR_TRAIN_TIME = 50
DYNAMIC_TRAIN_TIME = 10

# reward discount
# TAU = 0.01      # soft replacement
TRAIN_TIME = 300
MEMORY_CAPACITY = 15000
BATCH_SIZE = 32
ROLL_OUTS = 20  # PI2并行采样数
SAMPLE_SIZE = 128  # 训练时采样数，分成minibatch后进行训练
ENV_NAME = "InvertedDoublePendulum-v1"
PI2_coefficient = 30
MINI_BATCH = 1 # 训练的时候的minibatch
NUM_EPISODES = 2  # 每次rollout_train采样多少条轨迹
load_model_path = './data_0716/models.ckpt'
save_model_path = './data_op/models.ckpt'
"""
=========================流程==================================
self.learn()函数包含一次采样（rollout_train）和一次训练（update）
rollout_trian函数使用self.pi2_critic函数选取动作，这个函数输入状态，根据actor产生动作，然后使用PI2的方法产生合成动作
update分为critic update和action update。
"""


class PI2_Critic(object):
    def __init__(self, a_dim, s_dim, a_bound, env=None, buffer=None):
        self.dynamic_memory = np.zeros((MEMORY_CAPACITY, s_dim + s_dim + a_dim), dtype=np.float32)
        # 1(the last dimension) for reward
        self.num_episodes = NUM_EPISODES
        self.minibatch = MINI_BATCH
        self.sample_size = SAMPLE_SIZE
        self.trainfromscratch = TRAIN_FROM_SCRATCH
        self.sess = tf.Session()
        self.env = copy_env(env)
        self.reset_env = copy_env(env)
        self.globaltesttime = 0
        self.vtrace_losses = []
        self.dlosses = []
        self.alosses = []
        self.dynamic_model = Dynamic_Net(s_dim, a_dim,'dm')
        if buffer == None:
            self.buffer = Replay_buffer()
        else:
            self.buffer = buffer
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        if not self.trainfromscratch:
            self.buffer.load_data()



    def sample_ddpg(self, episodes):
        episodes_states = []
        episodes_actions = []
        episodes_rewards = []
        episodes_nstates= []
        for episode in episodes:
            # 输入一串episode，该函数会按顺序返回states，actions，reward， probability
            epi = copy.deepcopy(episode)
            length = len(epi) - 1
            states = np.zeros([length, self.s_dim])
            actions = np.zeros([length, 1])
            rewards = np.zeros([length,1])
            next_states = np.zeros([length, self.s_dim])


            for i in range(length):
                pair = copy.deepcopy(epi[i])
                state = pair[:self.s_dim]
                action= pair[self.s_dim:self.s_dim + self.a_dim]
                reward = pair[-self.s_dim - 1:-self.s_dim]
                next_state = pair[self.s_dim + self.a_dim + 1: self.s_dim + self.a_dim + 1 + self.s_dim]

                states[i] = state
                actions[i] = action
                rewards[i] = reward
                next_states[i] = next_state


            episodes_states.append(copy.deepcopy(states))
            episodes_actions.append(copy.deepcopy(actions))
            episodes_rewards.append(copy.deepcopy(rewards))
            episodes_nstates.append(copy.deepcopy(next_states))

        return episodes_states, episodes_actions, episodes_rewards, episodes_nstates
