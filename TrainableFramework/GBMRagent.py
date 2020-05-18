'''
构建智能体类

包含两个部分，一个是init 部分，对各个用到的模块进行实例化；
另外一部分是功能函数的构建，根据我们实际交互的需求，用实例化之后的模块构建相应的功能；
供交互过程调用。

'''
import EncoderDecoder
import random
import tensorflow as tf
import Controller
import MemReadWrite
import Memory
import networkx as nx
import MemReconstruction
from functools import reduce

class Agent():
    def __init__(self,num_actions=None,dim_obs=None,memory_size=100,memory_word_size=32,name="TrainableAgent"):
        '''
        智能体对环境的基本认知，动作空间，状态空间
        '''
        self.num_actions = num_actions
        self._obs_size = reduce(lambda x,y: x*y, list(dim_obs)) 
        self.memory_size = memory_size
        self.memory_word_size=memory_word_size
        '''
        1.实例化各个类
            五个模块的引入
        '''
        # 编解码器的实例化
        self._im2state = EncoderDecoder.ImEncoder(self._obs_size,self.memory_word_size,64)
        self._state2im = EncoderDecoder.ImDecoder(self._obs_size,self.memory_word_size,64)
        self._vae = EncoderDecoder.VAE(self._obs_size,self.memory_word_size,64)    
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self._vdecoder = EncoderDecoder.VDecoder(1,200)# 因为没用batch 所以是一维输出，就是一个v值
        self._vaev = EncoderDecoder.VAEV(self._obs_size,self.memory_word_size,64)
        # 控制器实例化
        self._controllercore = Controller.ControllerCore(num_actions=self.num_actions,
                                                        num_node=10,
                                                        memory_size=self.memory_size,
                                                        memory_word_size=self.memory_word_size)
        self._abstract_memory= self._controllercore.AbstractG
        self._external_memory = Memory.ExternalMemory(memory_size=self.memory_size)
        self._aggregator = Controller.MeanAggregator(self.memory_word_size, self.memory_word_size, name="aggregator", concat=False)#输入输出维度相同


        # 读写器实例化
        memory_num_reads = 2
        memory_top_k = 3
        self._memory_reader = MemReadWrite.MemReader(
            memory_word_size=self.memory_word_size,#这个与隐藏层同维度
            num_read_heads=memory_num_reads,
            top_k=memory_top_k,
            memory_size=self.memory_size)

        self._memory_eraser =MemReadWrite.MemErase(
            memory_word_size=self.memory_word_size,
            memory_size=self.memory_size)

        self._memory_writer = MemReadWrite.MemWriter(            
            memory_word_size=self.memory_word_size,
            memory_size=self.memory_size)        
        
        #重构机制实例化
        self._memory_reconstructor = MemReconstruction.MemReconstructor(memory_word_size=self.memory_word_size)
        
    '''
    2.构建功能函数
    '''
    def TakeRandomAction(self):
        action = random.randint(0,self.num_actions-1)
        return action

    # 编解码功能
    def obs2state(self,observation):
        obs = observation.reshape(1,self._obs_size).astype('float32') / 255 #这个应该放到函数里面
        obs_code = self._im2state(obs)
        return obs_code

    def state2obs(self,state):
        reconstructed_obs = self._state2im(state)
        return reconstructed_obs
    
    def state2value(self,state):
        value_estimate = self._vdecoder(state)
        return value_estimate

    # 轨迹记录 应该放在控制器中

    def EpsHistory_Initial(self):
        #目前是直接存储每个回合的数据，后续可以加入一个lstm的循环
        self._epshistory = []
        return self._epshistory

    def EpsHistory_add(self,state):
        self._epshistory.append(state)
        # 可以尝试对当前episode 的历史做深加工
        return self._epshistory

    #前向推断
    def infer(self,state,epshistory):
        '''
        功能：
        根据当前状态和本回合历史，从抽象图中查询相似与相关联的关键状态；（controller中完成）
        以关键状态为索引，在记忆模块中读取相应状态；（MemReadWrite）
        根据读取的相应状态及对应策略，做出决策（controller）

        输入：
        state : 当前状态
        epshistory: 当前episode 历史

        输出：
        
        action ： 决策
        '''
        if self._controllercore.AbstractG.num_nodes_in_Mem()==0:#和下面那个if 可以合并
            action = self.TakeRandomAction()
            read_info = []
        else:
            read_info = self._controllercore.create_read_info(state,epshistory)#这个是从控制器中得到相应的读写索引
            if read_info:
                mem_reads = self._memory_reader.read_from_memory(read_info,self._external_memory)#而这个是根据读写索引从外存中得到候选策略
                action = self._controllercore.policynet(mem_reads)
            else:
                action = self.TakeRandomAction()

        
        return action,read_info 

    #记忆更新
    def Memory_update(self,epshistory):
        # 根据当前内存容量决定是否遗忘
        # 根据当前轨迹决定是否写入
        # 没有输出，直接更改agent的记忆模块

        # 参照DNC的逻辑
        # 先计算usage，再擦除，之后写入
        usage = random.randint(0,200)
        print("usage",usage)
        self._memory_eraser.memory_eraser(self._external_memory,usage)#可以直接在记忆模块上读写，没有返回值
        print("length",len(epshistory))
        self._memory_writer.memory_writer(self._external_memory,epshistory)


    # 记忆抽象
    def Memory_abstract(self):
        #从记忆图，得到，抽象图，更新控制器的过程
        # 在memreconstruction.py,输入是外部存储，得到的是控制器中的内部存储
        self._abstract_memory=self._controllercore.build_abstract_graph(self._external_memory)
        # 
        print("current abstract memory",self._abstract_memory.num_nodes_in_Mem())

        return True

    # 记忆重构
    def Memory_reconstruct(self):
        #根据抽象图来重新改变记忆的权值的过程。
        #输入是抽象图，或者待重构的起止点
        #直接更改记忆的结构，没有输出
        self._memory_reconstructor.reconstruct_by_abstract_graph(self._abstract_memory,self._external_memory)

        return True


    # 参数训练
    def NetInitialize(self):
        #TODO:
        #对各个待训练网络的参数进行初始化
        self.vae_initial()

    def train(self,input_train,epochs,batch_size):
        #TODO:
        #集合各个模块中出现的需要训练的参数
        pass

    def loss(self):
        pass


    def vae_initial(self):
        self._vae.compile(self._optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    def vaev_initial(self):
        self._vaev.compile(self._optimizer,
                            loss =[
                                tf.keras.losses.MeanSquaredError(),
                                tf.keras.losses.MeanSquaredError(),
                            ],
                            loss_weights=[1,0.2],
                            )

    def vae_train(self,input_train,epochs,batch_size):
        self._vae.fit(input_train, input_train, epochs=1, batch_size=64)
        self._vae.summary()



    def vaev_train(self,input_train,rewards,epochs,batch_size):
        #self._vaev.summary()
        target1 = input_train
        target2 = rewards
        self._vaev.fit(
            input_train, 
            [target1,target2],
             epochs=2, 
             batch_size=64,)
        self._vaev.summary()
    
    def train_agg(self,x,y):
        #在init中已经将网络实例化了，如果有标签就能直接用了
        self._aggregator.compile(self._optimizer, loss=tf.keras.losses.MeanSquaredError())
        self._aggregator.fit(x, y, epochs=1, batch_size=64)