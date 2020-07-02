'''
控制器模块

动态维护一个抽象图，作为控制器

待完成功能，根据当前状态生长树，并根据实际执行情况，不断地进行树剪枝

从原始图中抽象图， build abstract graph
根据当前状态，和轨迹，从抽象图中得到记忆索引 create read info，推理的重要部分
从记忆中读回推理并实施

'''

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
#import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import networkx as nx
import Memory
import utils
import numpy as np
import random
from tensorflow.keras import backend as K

class MeanAggregator(tf.keras.Model):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)


        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        
        self.dense_neights = layers.Dense(output_dim)
        self.dense_self = layers.Dense(output_dim)


        self.input_dim = input_dim
        self.output_dim = output_dim

    def aggwithoutpara(self, self_vecs, neigh_vecs):

        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        self_vecs = tf.reduce_mean(self_vecs,axis=1)#这个只是为了匹配维度之前是（n,1,32）,现在变成（n，32）
       
        # [nodes] x [out_dim]
        # from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_neighs = neigh_means #先不考虑参数训练
        # from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        from_self = self_vecs
        
        if not self.concat:
            # print("we did not concat")
            # print("the size of from self",from_self)
            # print("the seize of from neighs",from_neighs)
            output = tf.add_n([from_self, from_neighs])
        else:
            # print("we did concat")
            # print("the size of from self",from_self)
            # print("the seize of from neighs",from_neighs)
            output = tf.concat([from_self, from_neighs], axis=1)
        
        # bias
        # if self.bias:
        #     output += self.vars['bias']
       
        return self.act(output)

    def call(self, self_vecs, neigh_vecs):
        self_vecs = tf.reduce_mean(self_vecs, axis=1)
        neigh_vecs = tf.reduce_mean(neigh_vecs, axis=1)
        neigh_means = neigh_vecs
        self_vecs = self_vecs#tf.reduce_mean(self_vecs,axis=1)#这个只是为了匹配维度之前是（n,1,32）,现在变成（n，32）
       
        # [nodes] x [out_dim]
        from_neighs = self.dense_neights(neigh_means)
        #from_neighs = neigh_means #先不考虑参数训练
        from_self = self.dense_self(self_vecs)
        #from_self = self_vecs
        output = tf.add_n([from_neighs,from_self])
        # if not self.concat:
        #     # print("we did not concat")
        #     # print("the size of from self",from_self)
        #     # print("the seize of from neighs",from_neighs)
        #     output = tf.add_n([from_self, from_neighs])
        # else:
        #     # print("we did concat")
        #     # print("the size of from self",from_self)
        #     # print("the seize of from neighs",from_neighs)
        #     output = tf.concat([from_self, from_neighs], axis=1)
        
        # bias
        # if self.bias:
        #     output += self.vars['bias']
       
        return self.act(output)    

    # def agg_loss(self,y):

class AggModel(tf.keras.Model):
    def __init__(self,memory_word_size=32,name='AggModel',**kwargs):
        super(AggModel, self).__init__(**kwargs)
        #目标是把两组输入，和一个aggregate 模型拼接起来
        #构造供训练的模型与loss
        self.memory_word_size=memory_word_size
        self._aggregator = MeanAggregator(self.memory_word_size, self.memory_word_size, name="aggregator", concat=False)
    
    # def affinity(self, inputs1, inputs2):
    #     """ Affinity score between batch of inputs1 and inputs2.
    #     Args:
    #         inputs1: tensor of shape [batch_size x feature_size].
    #     """
    #     # shape: [batch_size, input_dim1]
    #     if self.bilinear_weights:
    #         prod = tf.matmul(inputs2, tf.transpose(self.vars['weights']))
    #         self.prod = prod
    #         result = tf.reduce_sum(inputs1 * prod, axis=1)
    #     else: # 暂时使用这个，不进行新加参数
    #         result = tf.reduce_sum(inputs1 * inputs2, axis=1)
    #     return result
    
    def call(self,inputs):
        #print("inputs",inputs.shape())
        # input1=inputs[0]
        # neigh1=inputs[1]
        # input2=inputs[2]
        # neigh2=inputs[3]
        input1,neigh1,input2,neigh2 = inputs
        # print("input1 size",K.shape(input1))
        # print("input2 size",K.shape(input2))
        # print("neigh1 size",K.shape(neigh1))
        # print("neigh2 size",K.shape(neigh2))
        #print("inputs1",input1.shape())
        #print("neigh1",neigh1.shape())
        output1 = self._aggregator(input1,neigh1)
        output2 = self._aggregator(input2,neigh2)
        # 参考aggregate BipartiteEdgePredLayer 在predition py中
        # print("output1.size",K.shape(output1))
        # print("output2.size",K.shape(output2))
        aff = tf.reduce_sum(output1 * output2, axis=1)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        #print("true_xent",K.shape(true_xent))
        #losses = tf.reduce_sum(true_xent)#对多个维度而言，目前只有一个
        losses = true_xent
        #print("losses",K.shape(losses))
        return losses


class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits):

        # sample a random categorical action from given logits

        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class a2cModel(tf.keras.Model):

    def __init__(self, num_actions):

        super().__init__('mlp_policy')

        # no tf.get_variable(), just simple Keras API

        self.hidden1 = layers.Dense(128, activation='relu')

        self.hidden2 = layers.Dense(128, activation='relu')

        self.value = layers.Dense(1, name='value')

        # logits are unnormalized log probabilities

        self.logits = layers.Dense(num_actions, name='policy_logits')

        self.dist = ProbabilityDistribution()



    def call(self, inputs):

        # inputs is a numpy array, convert to Tensor

        x = tf.convert_to_tensor(inputs)

        # separate hidden layers from the same input tensor

        hidden_logs = self.hidden1(x)

        hidden_vals = self.hidden2(x)

        return self.logits(hidden_logs), self.value(hidden_vals)



    def action_value(self, obs):

        # executes call() under the hood

        logits, value = self.predict(obs)

        action = self.dist.predict(logits)

        # a simpler option, will become clear later why we don't use it

        # action = tf.random.categorical(logits, 1)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class ControllerCore(layers.Layer):
    def __init__(self,num_actions=4,num_node=10,memory_size=100,memory_word_size=32,name='ControllerCore'):
        # 外部存储的结构概览
        self.memory_size = memory_size
        self.memory_word_size = memory_word_size
        # 构造抽象图,从Memory中重新实例化一个子图，作为抽象图
        self.AbstractG = Memory.ExternalMemory(memory_size=self.memory_size)
        self.aggregator_cls = MeanAggregator
        self.aggregator = self.aggregator_cls(self.memory_word_size, self.memory_word_size, name="aggregator", concat=False)
        # 根据当前状态得到读取索引，用到相似度度量
        self._read_content_similarity = utils.CosineSimilarity(1,word_size=memory_word_size,name="read_content_similarity")
        # 策略输出
        self.epsilon =0.9
        self.num_actions=num_actions
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def run_random_walks(self,G, nodes, feature_size,num_walks):
        #在前向和反馈中均用到了 build abstract graph & train_agg
        pairs = []
        feature_matrix = np.zeros((len(nodes),num_walks,feature_size),dtype=np.float32)
        for count, node in enumerate(nodes):
            # print("now we turn on thecount  ",count)
            # print("now we turn on the node ",node)
            if G.degree(node) == 0:
                #print("empty neighbor for ",node)
                continue
            for i in range(num_walks):
                curr_node = node
                # print("curr_node is ",curr_node)
                for j in range(4):#walk len # 好像多了一轮没用的循环，
                    if len(G.neighbors(curr_node))==0:
                        # print("empty neighbor for ",curr_node)
                        continue
                    # print("the neighbors of curr_node",curr_node)
                    next_node = random.choice(G.neighbors(curr_node))
                    #print("the next node is ",next_node)
                    # self co-occurrences are useless
                    if curr_node != node:
                        #print("curr_node is not node it self, so we add feature in ",node,i,"with ",G.node[curr_node]['feature'])
                        pairs.append((node,curr_node))
                        feature_matrix[count,i,:]=G.node[curr_node]['feature']
                    else:
                        #print("curr_node is node it self")
                        pass
                    curr_node = next_node
            # if count % 10 == 0:
            #     print("Done walks for", count, "nodes")
        return pairs,feature_matrix

    def sagpooling(self, G,nodes,features,k):
        # G是我们的原图，nodes是所有节点，features是经过处理的节点特征矩阵，k是我们排序之后要取的部分节点的数目占比
        #完成对featrues矩阵的映射，映射到一个可排序的序列上，然后取出前kN个节点，组成新的图
        # 输出一个子图，可以只有节点没有边
        # 可以算个weight 乘到 features上，但是这个weight得能训练
        # 所以这里先求范数
        score = np.linalg.norm(features,ord=2,axis=1,keepdims=False)
        #print("score ",score)
        topk = tf.nn.top_k(score,int(k*len(nodes)))
        #print("topk.indices",topk.indices)
        #print("nodes",nodes)
        sub_nodes=[]
        for nodeid in topk.indices:
            sub_nodes.append(nodes[nodeid])
        #print("sub_nodes",sub_nodes)
        sub_graph = G.subgraph(sub_nodes).copy()
        #print("sub_graph",sub_graph)
        return sub_graph

    def build_abstract_graph(self,Ememory):
        # 参考sorb/search_policy.py和graphsage sagpooling
        #先把所有的都算了，但是实际上我们可以选择每次只计算episode的个数那么多
        self_vec =Ememory.node_feature_list()# 所有节点特征向量，这个比较容易得到，另外如果各个函数中对它都有需求，我们能不能在写入记忆的时候就把它完成？
        Gnodes = [n for n in Ememory.Gmemory.nodes()]
        #print("+++++++++++++++++++++++++++++",Gnodes)
        pairs,feature_matrix = self.run_random_walks(Ememory.Gmemory,Gnodes,self.memory_word_size,3) # 为每个节点找到2跳邻居 5个， 用节点序号做标记
        neigh_vecs = tf.cast(feature_matrix,tf.float32)#列为节点个数，行为邻居个数，每个元素为一个向量
        # aggregator()只是一层前向构造，接下来要用得到的特征进行重新采样，比如pooling 或者进行聚类的方式实现
        #outputs = self.aggregator.aggwithoutpara(self_vec,neigh_vecs)
        # self_vec = tf.reduce_mean(self_vec, axis=1)
        # neigh_vecs = tf.reduce_mean(neigh_vecs, axis=1)
        outputs = self.aggregator(self_vec,neigh_vecs)
        k = 0.2#我们要保留的节点比例
        self.AbstractG.Gmemory = self.sagpooling(Ememory.Gmemory,Gnodes,outputs,k)
        return self.AbstractG



    def create_read_info(self, state, history):
        # 根据历史和当前状态，从一个图结构self.AbstractG中找到相似节点，以及相关联的节点
        # 找相似节点的过程就是做图上的节点分类任务（找到相应的代码融合进来），并得到相应的相似度度量值，如果太大，就单独分类
        # 可以做的一个拓展就是，因为我们有了节点和历史，我们呢可以对子图抽象，然后找到结构相似的部分
        # 这里可以多读几条，然后在后续决策的过程中再做投票
        read_info = []
        #在abstractG中 找到和当前状态最相似的节点
        similarity = self._read_content_similarity.onevsall(state,self.AbstractG.node_feature_list())
        if max(similarity)<0:
            read_info =  [] #在abstract中没有找到相似的节点
        else:
            read_info = similarity.index(max(similarity))
            #他可能不止一个

        return read_info

    def policynet(self,state,mem_reads):
        # 根据读取到的量来生成策略
        # 不一定是训练的网络,所以用net 是不是不太合适
        action_candidates= mem_reads
        if action_candidates: 
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(action_candidates)
            else:
                # choose random action
                action = np.random.choice(list(range(self.num_actions))) 
        else:
            action = np.random.choice(list(range(self.num_actions)))      
        return action

    def policyModelfree(self,state):
        # 根据状态输出动作

        action = np.random.choice(list(range(self.num_actions)))      
        return action