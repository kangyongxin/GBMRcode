
# -*- coding: utf-8 -*-

# 在这里建立各中类型的策略网络

import os

import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
# 先搞个AC
# 在 Agent 里面进行实例化
# 加入引用源头

# 找到引用源头

class AC:
    '''
    离散动作的AC算法
    '''
    def __init__(self,statesize=None,num_actions=None):
        super(AC, self).__init__()
        self.statesize = statesize
        self.num_actions = num_actions
        # 调用的同时初始化actor 和critic 网络
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.gamma = 0.9 # 折扣因子

    def _build_critic(self):

        inputs = Input(shape=(self.statesize,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x)

        return model
    
    def _build_actor(self):

        inputs = Input(shape=(self.statesize,))
        x = Dense(20, activation='relu')(inputs)
        x = Dense(20, activation='relu')(x)
        x = Dense(self.num_actions, activation='sigmoid')(x)
        #s输出是每个动作的可能性
        model = Model(inputs=inputs, outputs=x)

        return model


    def _actor_loss(self,y_true,y_pred):
        '''
        构造actor网络的损失
        y_true 有两个值构成，第一个是动作，第二个是算出来的td error
        '''
        action_pred = y_pred
        action_true, td_error = y_true[:,0:-1],y_true[:,-1]
        print("action true",action_true)
        print("action_pred",action_pred)
        print("td",td_error)
        #action_true = K.reshape(action_true,(-1,1)) #action 做成n行1列的
        # onehot_act = to_categorical(action_true,num_classes=self.num_actions)
        # print("onehot act",onehot_act)
        #这里要把action——value 做成one hot 编码，然后和我们的pred 相比较
        #当我们用sigmoid函数作为神经元的激活函数时，最好使用交叉熵代价函数来替代方差代价函数，以避免训练过程太慢。
        #softmax与categorical_crossentropy
        loss = K.binary_crossentropy(action_true, action_pred)
        loss = loss * K.flatten(td_error)

        return loss
    
    def discount_reward(self, next_states, reward, done):
        """Discount reward for Critic

        Arguments:
            next_states: next_states
            rewards: reward of last action.
            done: if game done.
        """
        q = self.critic.predict(next_states)[0][0]

        target = reward
        if not done:
            target = reward + self.gamma * q

        return target

    def ACinitial(self):
        self.actor.compile(loss=self._actor_loss, optimizer=Adam(lr=0.001))
        self.critic.compile(loss='mse', optimizer=Adam(lr=0.01))        



# class A3C:
#     """A3C Algorithms with sparse action.
#     """
#     def __init__(self):
#         self.gamma = 0.95
#         self.actor_lr = 0.001
#         self.critic_lr = 0.01

#         self._build_model()
#         self.optimizer = self._build_optimizer()

#         # handle error
#         #self.sess = tf.InteractiveSession()
#         self.sess = tf.compat.v1.InteractiveSession()
#         tf.compat.v1.keras.backend.set_session(self.sess)
#         self.sess.run(tf.compat.v1.global_variables_initializer())

#     def _build_actor(self):
#         """actor model.
#         """
#         inputs = Input(shape=(4,))
#         x = Dense(20, activation='relu')(inputs)
#         x = Dense(20, activation='relu')(x)
#         x = Dense(1, activation='sigmoid')(x)

#         model = Model(inputs=inputs, outputs=x)

#         return model

#     def _build_critic(self):
#         """critic model.
#         """
#         inputs = Input(shape=(4,))
#         x = Dense(20, activation='relu')(inputs)
#         x = Dense(20, activation='relu')(x)
#         x = Dense(1, activation='linear')(x)

#         model = Model(inputs=inputs, outputs=x)

#         return model

#     def _build_model(self):
#         """build model for multi threading training.
#         """
#         self.actor = self._build_actor()
#         self.critic = self._build_critic()

#         # Pre-compile for threading
#         self.actor._make_predict_function()
#         self.critic._make_predict_function()

#     def _build_optimizer(self):
#         """build optimizer and loss method.

#         Returns:
#             [actor optimizer, critic optimizer].
#         """
#         # actor optimizer
#         actions = K.placeholder(shape=(None, 1))
#         advantages = K.placeholder(shape=(None, 1))
#         action_pred = self.actor.output

#         entropy = K.sum(action_pred * K.log(action_pred + 1e-10), axis=1)
#         closs = K.binary_crossentropy(actions, action_pred)
#         actor_loss = K.mean(closs * K.flatten(advantages)) - 0.01 * entropy

#         actor_optimizer = Adam(lr=self.actor_lr)
#         #actor_updates = actor_optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
#         actor_updates = actor_optimizer.get_updates(actor_loss,self.actor.trainable_weights)
#         # actor_train = K.function([self.actor.input, actions, advantages], [], updates=actor_updates)
#         actor_train = K.function([self.actor.input], [actions, advantages], updates=actor_updates)

#         # critic optimizer
#         discounted_reward = K.placeholder(shape=(None, 1))
#         value = self.critic.output

#         critic_loss = K.mean(K.square(discounted_reward - value))

#         critic_optimizer = Adam(lr=self.critic_lr)
#         #critic_updates = critic_optimizer.get_updates(self.critic.trainable_weights, [], critic_loss)
#         critic_updates = critic_optimizer.get_updates(critic_loss,self.critic.trainable_weights)
#         # critic_train = K.function([self.critic.input, discounted_reward], [], updates=critic_updates)
#         critic_train = K.function([self.critic.input], [discounted_reward], updates=critic_updates)

#         return [actor_train, critic_train]

#     def A3Cinitial(self):
#         pass