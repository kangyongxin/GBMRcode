
# 定义控制器的核心模块

#一个抽象图的相关操作，仿照RMAcore完成

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import networkx as nx

'''
规定抽象图的大小，也就是节点个数，聚类中心的个数

'''
class ControllerCore(layers.Layer):
    def __init__(self,num_node=10,name='ControllerCore'):
        self.AbstractG = self._build_graph()


    #抽象图的构建要由重构模块根据当前的记忆进行抽象，不一定放在这个位置，
    #刚开始的那一轮迭代中，原始图，抽象图都是空的
    def _build_graph(self):
        # 参考sorb/search_policy.py
        g = nx.DiGraph()
        # 建立一个图，还需要边权信息
        #pdist_combined = np.max(pdist, axis=0)
        #然后逐一将记录写入
        # for i, s_i in enumerate(rb_vec):
		# 	for j, s_j in enumerate(rb_vec):
		# 		length = pdist_combined[i, j]
		# 		if length < self._agent._max_search_steps:
		# 			g.add_edge(i, j, weight=length)
        return g

    def create_read_info(self, state, history):
        # 根据历史和当前状态，从一个图结构self.AbstractG中找到相似节点，以及相关联的节点
        # 找相似节点的过程就是做图上的节点分类任务（找到相应的代码融合进来），并得到相应的相似度度量值，如果太大，就单独分类
        # 可以做的一个拓展就是，因为我们有了节点和历史，我们呢可以对子图抽象，然后找到结构相似的部分
        # 这里可以多读几条，然后在后续决策的过程中再做投票
        read_info = []
        return read_info

    def policynet(self,mem_reads):
        # 根据读取到的量来生成策略
        # 不一定是训练的网络,所以用net 是不是不太合适

        action = mem_reads
        return action