#参考 https://geektutu.com/post/tensorflow2-gym-dqn.html


from random import randint
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from collections import deque
import random 

class DQNAgent():
    def __init__(
        self,n_actions,
        n_features,
        learning_rate= 0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=300,
        memory_size=2000,
        batch_size=256,
        e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate # 学习率，用来更新网络参数
        self.gamma = reward_decay # 折扣因子
        self.epsilon_max = e_greedy # 贪心算法
        self.replace_target_iter = replace_target_iter #更新目标网络参数的间隔
        self.memory_size = memory_size# 存储空间
        self.batch_size = batch_size# 每次训练的batch大小
        self.epsilon_increment = e_greedy_increment# 增量式调节我们的探索率
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        #如果是逐渐增加那么就从零开始，否者使用固定的
        self.learn_step_counter = 0# 记录学习步骤
        self.memory_deque = deque(maxlen=self.memory_size)# 初始化500个[s, a, r, s_]
        self._build_net()



    def _build_net(self):
        
        # build evaluate net
        #输入是当前状态 维度是n_feature，这里没有使用卷积，只是用全连接；
        # 输出是对每个动作的Q值的估计，目前还是针对离散动作
        #用keras 构造一个前向网络
        eva_model = tf.keras.Sequential()
        eva_model.add(tf.keras.layers.Dense(256,input_dim=self.n_features,activation='tanh'))#这里的128是隐藏节点的个数，找找其它代码的经验值
        eva_model.add(tf.keras.layers.Dense(128,activation='tanh'))
        eva_model.add(tf.keras.layers.Dense(self.n_actions,activation='linear'))
        
        eva_model.compile(loss='mse',optimizer=tf.keras.optimizers.RMSprop(lr=self.lr))
        self.eva_model = eva_model
        # build target net 
        tar_model = tf.keras.Sequential()
        tar_model.add(tf.keras.layers.Dense(256,input_dim=self.n_features,activation='tanh'))#这里的128是隐藏节点的个数，找找其它代码的经验值
        tar_model.add(tf.keras.layers.Dense(128,activation='tanh'))
        tar_model.add(tf.keras.layers.Dense(self.n_actions,activation='linear'))
        
        tar_model.compile(loss='mse',optimizer=tf.keras.optimizers.RMSprop(lr=self.lr))
        self.tar_model = tar_model



    def store_transition(self,s,a,r,s_):
        # r 可以在这里做重新修饰
        self.memory_deque.append((s,a,s_,r))


    def choose_action(self,observation):
        #这个要根据epsilon 贪心来选择策略
        if np.random.uniform()< self.epsilon:
            return np.argmax(self.eva_model.predict(observation))
        else:
            return self.take_random_action()
        # 刚开始时，加一点随机成分，产生更多的状态
        # if np.random.uniform() < epsilon - self.step * 0.0002:
        #     return np.random.choice([0, 1, 2])
        # return np.argmax(self.model.predict(np.array([s]))[0])

    def take_random_action(self):
        action = randint(0,self.n_actions-1)
        return action

    def save_model(self,filepath):
        print("model is saved in ", filepath)
        self.eva_model.save(filepath)

    def train_model(self):
        if len(self.memory_deque)<self.memory_size:
            #还没满
            return
        #print("training")
        self.learn_step_counter +=1 
        if self.learn_step_counter % self.replace_target_iter == 0:
            # tf 2.0中一个网络给另外一个网络复制参数，
            print("rewrite tar para")
            self.tar_model.set_weights(self.eva_model.get_weights())
        
        replay_batch = random.sample(self.memory_deque,self.batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])
        s_batch = np.squeeze(s_batch)
        next_s_batch = np.squeeze(next_s_batch)

        Q = self.eva_model.predict(s_batch)
        Q_next = self.tar_model.predict(next_s_batch)

        for i,replay in enumerate(replay_batch):
            _,a,_,reward = replay
            Q[i][a] = (1-self.lr)*Q[i][a]+ self.lr*(reward + self.gamma * np.amax(Q_next[i]))

        self.eva_model.fit(s_batch,Q,verbose=0)