
# 定义控制器的核心模块

#一个抽象图的相关操作，仿照RMAcore完成

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import networkx as nx
import Memory
import utils
import numpy as np
import random

'''
规定抽象图的大小，也就是节点个数，聚类中心的个数

'''
class ControllerCore(layers.Layer):
    def __init__(self,num_actions=4,num_node=10,memory_size=100,memory_word_size=100,name='ControllerCore'):
        #self.AbstractG = self._build_graph()
        #由于目前还没有构造抽象图 20200430
        #所以用原图代替
        self.memory_size = memory_size
        self.AbstractG = Memory.ExternalMemory(memory_size=self.memory_size)
        self._read_content_similarity = utils.CosineSimilarity(1,word_size=memory_word_size,name="read_content_similarity")
        self.epsilon =0.9
        self.num_actions=num_actions
        self.aggregator_cls = MeanAggregator
        self.memory_word_size = memory_word_size
        self.aggregator = self.aggregator_cls(self.memory_word_size, self.memory_word_size, name="aggregator", concat=False)

    #多层的时候要用aggregate进行封装
    def aggregate1(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        # 应该是在分层的时候有用，我们暂时先来一层试试，可能用不到这么多
        # 直接把meanaggregate类实例化，然后送到build函数中用
        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators
    #抽象图的构建要由重构模块根据当前的记忆进行抽象，不一定放在这个位置，
    #刚开始的那一轮迭代中，原始图，抽象图都是空的

    def run_random_walks(self,G, nodes, feature_size,num_walks):
        pairs = []
        feature_matrix = np.zeros((len(nodes),num_walks,feature_size),dtype=np.float32)
        for count, node in enumerate(nodes):
            print("now we turn on the node ",node)
            if G.degree(node) == 0:
                print("empty neighbor for ",node)
                continue
            for i in range(num_walks):
                curr_node = node
                print("curr_node is ",curr_node)
                for j in range(4):#walk len # 好像多了一轮没用的循环，
                    if len(G.neighbors(curr_node))==0:
                        print("empty neighbor for ",curr_node)
                        continue
                    print("the neighbors of curr_node",curr_node)
                    next_node = random.choice(G.neighbors(curr_node))
                    print("the next node is ",next_node)
                    # self co-occurrences are useless
                    if curr_node != node:
                        print("curr_node is not node it self, so we add feature in ",node,i,"with ",G.node[curr_node]['feature'])
                        pairs.append((node,curr_node))
                        feature_matrix[node,i,:]=G.node[curr_node]['feature']
                    else:
                        print("curr_node is node it self")
                    curr_node = next_node
            if count % 1000 == 0:
                print("Done walks for", count, "nodes")
        return pairs,feature_matrix
    def sagpooling(self, G,nodes,features,k):
        # G是我们的原图，nodes是所有节点，features是经过处理的节点特征矩阵，k是我们排序之后要取的部分节点的数目占比
        #完成对featrues矩阵的映射，映射到一个可排序的序列上，然后取出前kN个节点，组成新的图
        # 输出一个子图，可以只有节点没有边
        # 可以算个weight 乘到 features上，但是这个weight得能训练
        # 所以这里先求范数
        score = np.linalg.norm(features,ord=2,axis=1,keepdims=False)
        print("score ",score)
        topk = tf.nn.top_k(score,int(k*len(nodes)))
        print("topk.indices",topk.indices)
        print("nodes",nodes)
        sub_nodes=[]
        for nodeid in topk.indices:
            sub_nodes.append(nodes[nodeid])
        print("sub_nodes",sub_nodes)
        sub_graph = G.subgraph(sub_nodes).copy()
        print("sub_graph",sub_graph)
        return sub_graph

    def build_abstract_graph(self,Ememory):
        # 参考sorb/search_policy.py和graphsage
        # 前向构图的过程，和反馈训练的过程
        #目标：self.AbstractG.Gmemory = f(Ememory.Gmemory)
        # 先使用固定组合权重，用一个均匀分布的W表示，后期再考虑什么时机训练这个权重
        # 找到graphsage中前向运算的部分
        #先把所有的都算了，但是实际上我们可以选择每次只计算episode的个数那么多
        self_vec =Ememory.node_feature_list()# 所有节点特征向量，这个比较容易得到，另外如果各个函数中对它都有需求，我们能不能在写入记忆的时候就把它完成？
        Gnodes = [n for n in Ememory.Gmemory.nodes()]
        pairs,feature_matrix = self.run_random_walks(Ememory.Gmemory,Gnodes,self.memory_word_size,3) # 为每个节点找到2跳邻居 5个， 用节点序号做标记
        
        #print("------------self_vecs",self_vec)
        # print("-------------pairs",pairs)
        # print("-------------features",feature_matrix[0,:,:])
        neigh_vecs = tf.cast(feature_matrix,tf.float32)
        #neigh_vecs= self.get_neigh_vecs()
        #hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples] 想办法重新构造一批关于邻居的输入
        # 列为节点个数，行为邻居个数，每个元素为一个向量

        #train_adj_info = tf.assign(adj_info, minibatch.adj)
        #adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) #选取一个张量中索引对应的元素
        # adj是在构造邻接矩阵，而我们要找的是邻居节点对应的向量
        # 核心问题就是我们要把几个邻居的向量整合起来，共给aggregate函数使用

        #neigh_vecs = 采样几个邻居的特征向量，具体形式是啥样的？这个量的得到还需要一个辅助的索引，也就是我们要用随机游走的方式把邻居事先找到
        # graphsage中utils.py run_random_walk 似乎是做这件事情的
        #定义aggregator():输入是self_vec,neigh_vecs,两组向量，先按照graphsage中的meanaggregator来构造
        # aggregator()只是一层前向构造，接下来要用得到的特征进行重新采样，比如pooling 或者进行聚类的方式实现
        outputs = self.aggregator.aggwithoutpara(self_vec,neigh_vecs)
        print("shuchu ",outputs)
        k=0.5 #我们要保留的节点个数
        #这个output看作我们在sagpool中的z,取k个作为中心节点，按照什么原则取？如果跟邻域长得越像，那么我们得到的乘积会越大，前提是我们归一化所有向量，因为邻域的内积相当于向量夹角
        #根据outputs对每个节点计算一个跑分，
        #只要返回重要节点的序号即可，我们不打算在缩略图中计算边
        #还要考虑如何把节点特征转换为向量，先用模值来实现，原文中加了一个N维参数，N是节点个数，可是这里节点个数不确定，所以就没法进行训练了
        #如果能从一个图中得到一个子图就更好了
        self.AbstractG.Gmemory = self.sagpooling(Ememory.Gmemory,Gnodes,outputs,k)

        print("---------------sub graph nodes",self.AbstractG.num_nodes_in_Mem())
        print("---------------nodes features",self.AbstractG.node_feature_list())
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

    def policynet(self,mem_reads):
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

class MeanAggregator(layers.Layer):
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
            print("we did not concat")
            print("the size of from self",from_self)
            print("the seize of from neighs",from_neighs)
            output = tf.add_n([from_self, from_neighs])
        else:
            print("we did concat")
            print("the size of from self",from_self)
            print("the seize of from neighs",from_neighs)
            output = tf.concat([from_self, from_neighs], axis=1)
        
        # bias
        # if self.bias:
        #     output += self.vars['bias']
       
        return self.act(output)