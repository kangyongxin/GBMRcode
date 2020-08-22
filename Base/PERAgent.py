#参考 https://geektutu.com/post/tensorflow2-gym-dqn.html

# 在DQNAgent的基础上完成
from random import randint
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from collections import deque
import random 


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory():
    def __init__(self, capacity=100):
        """
        This Memory class is modified based on the original code from:
        https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
        """
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error  
        self.tree = SumTree(capacity)    

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

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
        #self.memory_deque = deque(maxlen=self.memory_size)# 初始化500个[s, a, r, s_]
        self.memory = Memory(capacity=self.memory_size)
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
        # self.memory_deque.append((s,a,s_,r))
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)    # have high priority for newly arrived transition

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