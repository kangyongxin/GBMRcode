# graph based memory reconstruction agents
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Canopy import Canopy as Cluster
import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
#import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from tensorflow.keras import backend as K
import random
from CanopyDetail import Canopy as Cluster
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin,pairwise_distances_argmin_min



class MeanAggregator(tf.keras.Model):
    """
    Aggregates via mean followed by matmul and non-linearity.
    这是个输出的类
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
        self_vecs = self_vecs# tf.reduce_mean(self_vecs,axis=1)#这个只是为了匹配维度之前是（n,1,32）,现在变成（n，32）
       
        # [nodes] x [out_dim]
        # from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_neighs = neigh_means #先不考虑参数训练
        # from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        from_self = self_vecs
        
        if not self.concat:
            #output = tf.add_n([from_self, from_neighs])
            #output = tf.subtract(from_self, from_neighs)
            output = from_neighs
        else:
            output = tf.concat([from_self, from_neighs], axis=1)
       
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

        return self.act(output)    

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
        print("input1 size",K.shape(input1))
        print("input2 size",K.shape(input2))
        print("neigh1 size",K.shape(neigh1))
        print("neigh2 size",K.shape(neigh2))
        #print("inputs1",input1.shape())
        #print("neigh1",neigh1.shape())
        output1 = self._aggregator(input1,neigh1)
        output2 = self._aggregator(input2,neigh2)
        # 参考aggregate BipartiteEdgePredLayer 在predition py中
        print("output1.size",K.shape(output1))
        print("output2.size",K.shape(output2))
        aff = tf.reduce_sum(output1 * output2, axis=1)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        print("true_xent",K.shape(true_xent))
        #losses = tf.reduce_sum(true_xent)#对多个维度而言，目前只有一个
        losses = true_xent
        print("losses",K.shape(losses))
        return losses


#learning_rate=0.01, reward_decay=0.9, e_greedy=0.97
class GMRAgent:
    def __init__(self, actions,e_greedy=0.97,gamma=0.9,lr=0.01):
        self.actions=actions
        #print(actions)
        self.epsilon=e_greedy
        self.gamma = gamma
        self.lr = lr
        self.node_vec=[]
        self.Gmemory=nx.DiGraph()
        self.StateAttributesDict={}
        self.StateLabelDict= {}
        self.Centers=[]
        self.reconstruct_paths=[]
        self.memory_word_size = 4
        self.aggregator_cls = MeanAggregator
        # 输入和输出的维度都是4
        self.aggregator = self.aggregator_cls(self.memory_word_size, self.memory_word_size, name="aggregator", concat=False)
        



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

    def random_action(self,state):
        action = np.random.choice(self.actions)
        return action

    def MemoryWriter(self, re_vec):
        '''
        输入是一条轨迹，先把每个状态上的值算出来
        '''

        # G = 0 
        for i in range(len(re_vec)):
        #for i in range(len(re_vec)-1,-1,-1):
           
            self.PairWriter(re_vec[i][0],re_vec[i][1],re_vec[i][2],re_vec[i][3])

    def update_edge(self,state, action, w, state_):
        old_w= self.Gmemory.edges[state,state_]['weight']
        old_visits = self.Gmemory.edges[state,state_]['visits']
        if state_ != 'terminal':
            edge_list = self.Gmemory.edges(state_)
            if len(edge_list)==0:
                #下一个状态既不是终点
                tar = w
            else:
                weight_list = []
                for edge_i in edge_list:
                    weight_list.append(self.Gmemory.edges[edge_i]['weight'])
                weight_max = np.max(weight_list)
                tar =  w + self.gamma*weight_max
        else:
            tar = w
        delta_w = tar - old_w
        new_w = old_w + self.lr*delta_w
        self.Gmemory.add_edge(state,state_,weight=new_w,labels=action,reward=w,visits=old_visits+1)

    def PairWriter(self, state, action, w, state_):
        '''
        节点是状态，用一个列表表示（状态本身的特征需要另外表示，比如，出现次数，距离远近，用轨迹算得到的值函数）
        边是动作转移（边的特征可能会有很多，比如，出现次数，状态动作值函数，针对确定性环境，一个状态动作只能对应下一个状态动作）
        (这里只记一步可达状态作为边，权重先用reward)
        '''
        if [state,state_] in self.Gmemory.edges():
            #修边，用state_对应的边权来修正state,action的比安全
            self.update_edge(state,action,w,state_)
        else:
            #加边
            if self.check_state_exist(state):
                if self.check_state_exist(state_):
                    pass
                else:
                    self.Gmemory.add_node(state_,attributes=self.StateAttributesDict[state_],label=self.StateLabelDict[state_])
            else:
                if self.check_state_exist(state_):
                    self.Gmemory.add_node(state,attributes=self.StateAttributesDict[state],label=self.StateLabelDict[state])
                else:
                    self.Gmemory.add_node(state,attributes=self.StateAttributesDict[state],label=self.StateLabelDict[state])
                    self.Gmemory.add_node(state_,attributes=self.StateAttributesDict[state_],label=self.StateLabelDict[state_])
            self.Gmemory.add_edge(state,state_,weight=w,labels=action,reward=w,visits=1)

    def plotGmemory(self):
        #print('节点向量的长度',len(self.node_vec))
        print("输出全部节点：{}".format(self.Gmemory.number_of_nodes()))
        # print("输出全部边：{}".format(self.Gmemory.edges()))
        print("输出全部边的数量：{}".format(self.Gmemory.number_of_edges()))
        pos = {}#nx.spring_layout(self.Gmemory)  # positions for all nodes
        labels = {}
        for node in list(self.Gmemory.nodes()):
            #pos[node] = (math.floor(self.Gmemory.nodes[node]['label']/10),10-self.Gmemory.nodes[node]['label']%10)
            pos[node] = (self.Gmemory.nodes[node]['label']%10,10-math.floor(self.Gmemory.nodes[node]['label']/10))
            labels[node] = self.Gmemory.nodes[node]['label']
            #print(node,pos[node],labels[node])
        #print(pos,labels)
        nx.draw_networkx(self.Gmemory,pos,with_labels=False,node_size=50)
        # nx.draw_networkx_labels(self.Gmemory, pos, labels, font_size=4, font_weight='bold')
        # nx.draw_networkx_nodes(self.Gmemory,pos,nodelist=self.Centers,with_labels=False,node_size=100,node_color='r')
        plt.axis("off")
    def plotAbastract(self):
        pos = {}#nx.spring_layout(self.Gmemory)  # positions for all nodes
        labels = {}
        for node in list(self.Gmemory.nodes()):
            #pos[node] = (math.floor(self.Gmemory.nodes[node]['label']/10),10-self.Gmemory.nodes[node]['label']%10)
            pos[node] = (self.Gmemory.nodes[node]['label']%10,10-math.floor(self.Gmemory.nodes[node]['label']/10))
            labels[node] = self.Gmemory.nodes[node]['label']
        elist = []
        for ni in self.Centers:
            for nj in self.Centers:
                if nx.algorithms.shortest_paths.generic.has_path(self.Gmemory,ni,nj):
                    elist.append((ni,nj))
        nx.draw_networkx_edges(self.Gmemory, pos, edgelist=elist)
        nx.draw_networkx_labels(self.Gmemory, pos, labels, font_size=4, font_weight='bold')
        nx.draw_networkx_nodes(self.Gmemory,pos,node_size=100) 
        nx.draw_networkx_nodes(self.Gmemory,pos,nodelist=self.Centers,with_labels=True,node_size=100,node_color='r')
        plt.axis("off")

    def plotReconPath(self):
        pos = {}#nx.spring_layout(self.Gmemory)  # positions for all nodes
        labels = {}
        for node in list(self.Gmemory.nodes()):
            #pos[node] = (math.floor(self.Gmemory.nodes[node]['label']/10),10-self.Gmemory.nodes[node]['label']%10)
            pos[node] = (self.Gmemory.nodes[node]['label']%10,10-math.floor(self.Gmemory.nodes[node]['label']/10))
            labels[node] = self.Gmemory.nodes[node]['label']
        elist = []
        for ni in self.Centers:
            for nj in self.Centers:
                if nx.algorithms.shortest_paths.generic.has_path(self.Gmemory,ni,nj):
                    path = nx.shortest_path(self.Gmemory,ni,nj)
                    for idx in range(len(path)-1):
                        pair_start = path[idx]
                        pair_end = path[idx+1]
                        elist.append((pair_start,pair_end))
        nx.draw_networkx_edges(self.Gmemory, pos, edgelist=elist)
        nx.draw_networkx_labels(self.Gmemory, pos, labels, font_size=4, font_weight='bold')
        nx.draw_networkx_nodes(self.Gmemory,pos,node_size=100) 
        nx.draw_networkx_nodes(self.Gmemory,pos,nodelist=self.Centers,with_labels=True,node_size=100,node_color='r')
        plt.axis("off")

    def plotFinalpath(self,finalpath):
        pos = {}#nx.spring_layout(self.Gmemory)  # positions for all nodes
        labels = {}
        for node in list(self.Gmemory.nodes()):
            #pos[node] = (math.floor(self.Gmemory.nodes[node]['label']/10),10-self.Gmemory.nodes[node]['label']%10)
            pos[node] = (self.Gmemory.nodes[node]['label']%10,10-math.floor(self.Gmemory.nodes[node]['label']/10))
            labels[node] = self.Gmemory.nodes[node]['label']
        elist= finalpath
        nx.draw_networkx_edges(self.Gmemory, pos, edgelist=elist,edge_color='g')
        nx.draw_networkx_labels(self.Gmemory, pos, labels, font_size=4, font_weight='bold')
        nx.draw_networkx_nodes(self.Gmemory,pos,node_size=100) 
        nx.draw_networkx_nodes(self.Gmemory,pos,nodelist=self.Centers,with_labels=True,node_size=100,node_color='r')
        plt.axis("off")     


    def MemoryReader(self, state):
        #与state一步相连的状态
        temp = self.Gmemory[state]
        next_state_candidates=list(temp)
        #print('next_state_candidates',next_state_candidates)   
        action_candidates=[]
        value_list =[]
        for i in range(len(next_state_candidates)):
            action_candidates.append(self.Gmemory.edges[state,next_state_candidates[i]]['labels'])
            value_list.append(self.Gmemory.edges[state,next_state_candidates[i]]['weight'])
        return action_candidates,value_list,next_state_candidates

    def check_state_exist(self,state):
        '''
        检查是否见过该状态，或者相近状态，如果没有，新加；有，返回中心值
        '''
        #print(self.Gmemory.node)
        if state in list(self.Gmemory.nodes):
            return True
        else:
            return False

    def get_action_value(self, state):
        '''
        根据图中节点找到可执行的边的权重??????
        同时还要加入未执行的边，零权重
        '''
        [action_candidates,value_list,next_state_candidates]=self.MemoryReader(state)
        for i in range(len(self.actions)):
            if self.actions[i] in action_candidates:
                pass
            else:
                action_candidates.append(self.actions[i])
                value_list.append(0)
        return action_candidates,value_list

    def ActAccordingToGM(self, state):
        '''
        在记忆库中查询state
        进行推演
        返回各个可能动作对应的值函数
        '''
        #print("actions",self.actions)
        if self.check_state_exist(state):
            #之前存在（相似状态），找到相应的值，按照贪心策略执行 
            if np.random.uniform() < self.epsilon:
                action_candidates,action_values= self.get_action_value(state)
                # some actions may have the same value, randomly choose on in these actions
                max_SA=np.max(action_values)
                acts=[]
                for i in range(len(action_values)):
                    if action_values[i]==max_SA:
                        acts.append(action_candidates[i])#这里宜直接使用动作的索引，而不是用i
                #print("candidates acts",acts)
                #print("action_values",action_values)
                action = np.random.choice(acts)
            else:
                # choose random action
                action = np.random.choice(self.actions)
        else:
            #之前不存在，
            action = np.random.choice(self.actions)
        return action

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
                for j in range(2):#walk len # 好像多了一轮没用的循环，
                    neis = list(G.neighbors(curr_node))#print("neighbors",list(G.neighbors(curr_node)))
                    if len(neis)==0:
                        # print("empty neighbor for ",curr_node)
                        continue
                    # print("the neighbors of curr_node",curr_node)
                    next_node = random.choice(neis)
                    #print("the next node is ",next_node)
                    # self co-occurrences are useless
                    if curr_node != node:
                        #print("curr_node is not node it self, so we add feature in ",node,i,"with ",G.node[curr_node]['feature'])
                        pairs.append((node,curr_node))
                        feature_matrix[count,i,:]=G.nodes[curr_node]['attributes']
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
        print("score ",score)
        topk = tf.nn.top_k(-score,int(k*len(nodes)))# 这里加了个符号，求最小的几个
        #print("topk.indices",topk.indices)
        #print("nodes",nodes)
        sub_nodes=[]
        for nodeid in topk.indices:
            sub_nodes.append(nodes[nodeid])
        #print("sub_nodes",sub_nodes)
        sub_graph = G.subgraph(sub_nodes).copy()
        #print("sub_graph",sub_graph)
        return sub_graph,sub_nodes


    def aggpool(self):
        # 得到节点特征
        #self_vec =Ememory.node_feature_list()# 理论上应该有函数可以直接完成
        self_vec=[]#dataset=[]
        for node in list(self.Gmemory.nodes()):
            self_vec.append(self.Gmemory.nodes[node]['attributes'])
        # print("self_cec",self_vec)

        # self_vec_matrix = np.array(self_vec)
        # self_vec_matrix3 = self_vec_matrix[:,np.newaxis]

        # print("self_vec",self_vec_matrix)
        # 得到所有节点的表示
        Gnodes = [n for n in self.Gmemory.nodes()]
        # 在图中进行随机游走
        n_walks= 4
        pairs,feature_matrix = self.run_random_walks(self.Gmemory,Gnodes,self.memory_word_size,n_walks) # 为每个节点找到2跳邻居 5个， 用节点序号做标记
        # print("feature_matrix",feature_matrix)# （n,n_walks,dim_obs）
        #delta_feature = feature_matrix - self_vec_matrix3
        #得到特征向量
        neigh_vecs = tf.cast(feature_matrix,tf.float32)#列为节点个数，行为邻居个数，每个元素为一个向量
        #neigh_vecs = tf.cast(delta_feature,tf.float32)
        
        # 先进行aggregate
        #print("selfvec",self_vec,"neigh_vecs",neigh_vecs)
        #outputs = self.aggregator(self_vec,neigh_vecs) #训练参数
        outputs = self.aggregator.aggwithoutpara(self_vec,neigh_vecs)# 无参数，加和平均
        print("output",outputs)
        # 再进行pooling
        # k=0.3
        # subgraph,center_list = self.sagpooling(self.Gmemory,Gnodes,outputs,k)
        # 最后得到子图的节点
        #center_list = subgraph.nodes()

        # 使用聚类的算法得到下采样
        FeatureDict =dict(zip(Gnodes,list(outputs)))
        
        #print("FeatureDict",keys)

        # for node in Gnodes:
        #     FeatureDict[node]
        t1 = 300
        t2 = 200
        self.gc = Cluster()#这个要在原函数上改改，不能直接放在这里
        self.gc.setThreshold(t1,t2)
        canopies = self.gc.clustering(FeatureDict)
        print("canopies",len(canopies))
        center_list=[]
        # for i in range(len(canopies)):
        #     center_list.append(canopies[i][0]) #这一步要把特征重新对应回标签上
        k_means = KMeans(n_clusters=len(canopies))
        k_means.fit_predict(outputs)
        k_means_cluster_centers = k_means.cluster_centers_
        print("k_means_cluster_centers",k_means_cluster_centers)
        argmin = pairwise_distances_argmin(k_means_cluster_centers,outputs,metric='euclidean')
        print("argmin",argmin)
        for t in argmin:
            center_list.append(Gnodes[t])
        
        return center_list



    def MemoryReconstruction(self,t1,t2):

        
        # # 0806 注释掉，换成GNN
        # dataset=[]
        # for node in list(self.Gmemory.nodes()):
        #     dataset.append(self.Gmemory.nodes[node]['attributes'])
        # #print("dataset",dataset)
        # gc = Cluster(dataset)
        # gc.setThreshold(t1,t2)
        # canopies = gc.clustering()
        # #print('Get %s initial centers.' % len(canopies))
        # center_list=[]
        # for i in range(len(canopies)):
        #     center_list.append(str(list(canopies[i][0])))
        
        # 0806 用GNN得到一个缩略图，不需要是全图，只要有关键节点就行，甚至连节点特征都不用
        # 输入是一幅图，（但是是每一步都变的图），输出是一个中心节点构成的列表
        center_list = self.aggpool()

        print("center list ",center_list)
        self.Centers = center_list
        print("reconstruct")
        for i in range(len(center_list)):
            # if center_list[i] in dataset:
            #     pass
            # else:
            #     print("where is the center i",center_list[i])
            for j in range(len(center_list)):
                if i==j:
                    pass
                else:
                    #print(center_list[i],center_list[j])
                    # if center_list[j] in dataset:
                    #     pass
                    # else:
                    #     print("where is the center j",center_list[j])
                    if nx.algorithms.shortest_paths.generic.has_path(self.Gmemory,center_list[i],center_list[j]):
                        # print("reconstruct")
                        path = nx.shortest_path(self.Gmemory,center_list[i],center_list[j])
                        #print("path",path)
                        self.reconstruct_paths.append(path)
                        temp=[]
                        for idx in range(len(path)-1):
                            pair_start = path[idx]
                            pair_end = path[idx+1]
                            temp.append([pair_start,self.Gmemory.edges[pair_start,pair_end]['labels'],self.Gmemory.edges[pair_start,pair_end]['reward'],pair_end])
                        #print("temp",temp)
                        self.MemoryWriter(temp)
                    else:
                        # print("no path")
                        pass
        

       
        
