# 一个对maze 和Atari环境都能适应的智能体结构
from functools import reduce
import random
from Memory import Memory
import numpy as np


class Agent():
    def __init__(self,num_actions=None,dim_obs=None,memory_size=100,memory_word_size=32,name="TrainableAgent"):
        '''
        智能体对环境的基本认知，动作空间，状态空间
        '''
        self.num_actions = num_actions
        # reduce (lambda x,y:x+y, [1,2,3]) 输出为 6
        self._obs_size = dim_obs
        self.memory_size = memory_size
        self.memory_word_size=memory_word_size
        self.StateAttributesDict={}
        self.StateLabelDict= {}
        self.ExternalMemory = Memory(self.memory_size)
    def TakeRandomAction(self):
        action = random.randint(0,self.num_actions-1)
        return action
    '''
    # 编码模块： 目前有三种，
    1 maze中是将观测的格子边界，转换为一个状态标记，用Dict 存储，同时把这个格子边界作为特征写到节点上，并打上一个序号标签，后面画图用到
    2 ram 的编码是有针对的，比如mspacman 就是把固定的维度拿出来做为位置特征， 这里的state 直接作为节点特征
    3 直接对图像进行编解码（未完成）

    '''
    def obs2state(self,observation):
        
        if observation == 'terminal':
            state= str(list([365.0,365.0,395.0,395.0])) #10*10的最后一个格子是多少
            # self.StateAttributesDict[state]=list([165.0,165.0,195.0,195.0])
            self.StateAttributesDict[state]=list([365.0,365.0,395.0,395.0])
            self.StateLabelDict[state]= 99
        else:
            state=str(observation)
            self.StateAttributesDict[state]=observation #为了把值传到后面重构部分进行计算
            self.StateLabelDict[state]=int(((observation[1] + 15.0 - 20.0) / 40) *10 + (observation[0] + 15.0 - 20.0) / 40 )
        return state

    def obs_ram(self,observation):
        #仅限于MsPacman
        state = np.array([[observation[10],observation[16]]])
        state = state.astype('float32')
        return state

    # # 编解码功能
    # def obs2state(self,observation):
    #     obs = observation.reshape(1,self._obs_size).astype('float32') / 255 #这个应该放到函数里面
    #     obs_code = self._im2state(obs)
    #     return obs_code

    # def state2obs(self,state):
    #     reconstructed_obs = self._state2im(state)
    #     return reconstructed_obs
    
    # def state2value(self,state):
    #     value_estimate = self._vdecoder(state)
    #     return value_estimate

