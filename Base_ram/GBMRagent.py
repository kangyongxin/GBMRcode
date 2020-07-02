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
import numpy as np
from tensorflow.keras import backend as K
#import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from Policynet import AC

class Agent():
    def __init__(self,num_actions=None,dim_obs=None,memory_size=100,memory_word_size=32,name="TrainableAgent"):
        '''
        智能体对环境的基本认知，动作空间，状态空间
        '''
        self.num_actions = num_actions
        # reduce (lambda x,y:x+y, [1,2,3]) 输出为 6
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

        self._aggregator = Controller.MeanAggregator(self.memory_word_size, self.memory_word_size, name="aggregator", concat=False)#输入输出维度相同
        self._aggmodel = Controller.AggModel(self.memory_word_size)
        self.batch_num =0 # 用来控制训练agg的batch起点，训练聚合器参数的时候用到
        #这一版的a2c写的不够清晰
        # self.a2cparams = {
        #     'gamma': 0.99,
        #     'value': 0.5,
        #     'entropy':0.0001
        # }
        # self.a2cmodel = Controller.a2cModel(num_actions=self.num_actions)
        # self.a2cmodel.compile(
        #     optimizer = ko.RMSprop(lr=0.0007),
        #     loss = [self._logits_loss,self._value_loss]
        # )
        self.a2cmodel = AC(self._obs_size,self.num_actions)
        # 存储器实例化
        self._abstract_memory= self._controllercore.AbstractG
        self._external_memory = Memory.ExternalMemory(memory_size=self.memory_size)

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
    
    def obs_ram(self,observation):
        #仅限于MsPacman
        state = np.array([[observation[10],observation[16]]])
        state = state.astype('float32')
        return state
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
                action = self._controllercore.policynet(state,mem_reads)
            else:
                action = self.TakeRandomAction()

        
        return action,read_info 

    def inferModelfree(self,state):
        action = self._controllercore.policyModelfree(state)
        return action
    
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
        #print("current abstract memory",self._abstract_memory.num_nodes_in_Mem())

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
        #self._vae.summary()



    def vaev_train(self,input_train,rewards,epochs,batch_size):
        #self._vaev.summary()
        target1 = input_train
        target2 = rewards  
        print("input_train shape",K.shape(input_train))  
        print("target2 ",K.shape(rewards))
        self._vaev.fit(
            input_train, 
            [target1,target2],
             epochs=2, 
             batch_size=64,)
        self._vaev.summary()
    

    def _inittrain(self):
        self.batch_num =0
        self.batch_size = 150
        edges = self._external_memory.Gmemory.edges()# 也可以用context pair
        self.train_edges = np.random.permutation(edges)

    def minibatch(self):
        # 从图的边中取出一部分（多少个？）
        # 得到想应的两端节点的特征
        # 作为训练 aggregate 的数据
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges)) 
        batch_edges = self.train_edges[start_idx : end_idx]  
        batch1 = []
        batch2 = []
        for node1,node2 in batch_edges:
            batch1.append(node1)
            batch2.append(node2)
        return batch1,batch2     
        
    def sample(self,inputs):
        #这是根据多层信息对邻居边界进行扩展的程序
        samples = [inputs]
        support_size = 1
        support_sizes = [support_size]
        
        return samples,support_sizes


    def train_agg(self):
        #在init中已经将网络实例化了，如果有标签就能直接用了
        self._inittrain()
        batch1,batch2 = self.minibatch()
        # 还要采样负例
        samples1,support_sizes1 = self.sample(batch1) #这个samples加s表示是一组，也就是我们是针对多层设计，这里目前只用一层
        samples2,support_sizes2 = self.sample(batch2)
        #print("samples ",samples1)
        pairs1,feat1 = self._controllercore.run_random_walks(self._external_memory.Gmemory,samples1[0],self.memory_word_size,3)
        pairs2,feat2 = self._controllercore.run_random_walks(self._external_memory.Gmemory,samples2[0],self.memory_word_size,3)
        inputs1 =  self._external_memory.read_features_from_nodes(samples1[0])
        inputs2 =  self._external_memory.read_features_from_nodes(samples2[0])
        inputs1 = tf.cast(inputs1,tf.float32) #[64,1,32]
        inputs2 = tf.cast(inputs2,tf.float32)
        neigh_vecs1 = tf.cast(feat1,tf.float32)#[64,3,32]
        neigh_vecs2 = tf.cast(feat2,tf.float32)
        # inputs1 = tf.reduce_mean(inputs1,axis= 1)
        # inputs2 = tf.reduce_mean(inputs2,axis=1)
        # neigh_vecs1 = tf.reduce_mean(neigh_vecs1,axis=1)
        # neigh_vecs2 = tf.reduce_mean(neigh_vecs2,axis=1)
        # 这是数据，是要fit 到函数的，现在还要构造模型
        #还要找到想应的邻居，才能把自己的值算出来，所以最好在sample中能把support 直接算出来
        # 尝试把loss 做成一个输出参考：https://www.spaces.ac.cn/archives/4493
        # def loss_function(y_true,y_pred):
        #     return y_pred
        y = [1 for i in range(len(inputs1))]
        y = tf.cast(y,tf.float32)
        y = tf.reshape(y,[len(inputs1),1])
        #print("y",y)
        #print("input1",inputs1)
        self._aggmodel.compile(self._optimizer, loss=lambda y_true,y_pred: y_pred)
        #几个输入的格式要一致
        # print("inputs1 shape",K.shape(inputs1))
        # print("inputs2 shape",K.shape(inputs2))
        # print("neigh_vec1 shape",K.shape(neigh_vecs1))
        # print("neigh_vec2 shape",K.shape(neigh_vecs2))
        # print("y shape",K.shape(y))
        self._aggmodel.fit([inputs1,neigh_vecs1,inputs2,neigh_vecs2], y, epochs=2, batch_size=64)
        # self._aggregator.compile(self._optimizer, loss=lambda y_true,y_pred: y_pred)
        # self._aggregator.fit([inputs1,neigh_vecs1,inputs2,neigh_vecs2], y, epochs=1, batch_size=64)
        # self._aggmodel.summary()
    
    
    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.a2cparams['gamma'] * returns[t+1] * (1-dones[t])
            #returns[t] = rewards[t] + self.a2cparams['gamma'] * returns[t+1] 
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.a2cparams['value']*kls.mean_squared_error(returns, value)



    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.a2cparams['entropy']*entropy_loss