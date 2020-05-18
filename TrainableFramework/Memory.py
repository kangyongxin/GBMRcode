'''
记忆存储模块
# 定义存储器
# 核心是一个有向图，对有向图进行一些包装，供其它函数使用
# 这里要做到可读写
所有读写内容均针对图完成，它是底层的数据结构，不能引入与智能体相关的内容
'''

import networkx as nx
import numpy as np

class ExternalMemory:
    def __init__(self, memory_size=100):
        self.Gmemory=nx.DiGraph()
        self.StateAttributesDict=[]
        self.memory_size = memory_size
        self._test_n =50
        #self.add_node_for_test()
        self.gamma = 0.9
        self.lr = 0.01

    def plotmemory(self):
        print("self._test_n",self._test_n)
        print("Gmemory.nodes", len(self.Gmemory))
        #print("feature",self.Gmemory.node[1]['feature'])

    def testadd(self,t):
        self._test_n += t

    def testminus(self,t):
        self._test_n -= t

    def num_nodes_in_Mem(self):
        return len(self.Gmemory)

    def num_edges_in_Mem(self):
        return self.Gmemory.size()

    def add_node_for_test(self):
        self.Gmemory.add_node(0,feature=[1,2,3],value=12.3)
        self.Gmemory.add_node(1,feature=[23,33,11],value=43.3)
        self.Gmemory.add_node(2,feature=[23,33,11],value=43.3)

    def check_node_exist(self,cur_node):
        if cur_node in self.Gmemory:
            return True
        else:
            return False

    def add_single_node(self,cur_node,cur_state,act,r):
        self.Gmemory.add_node(cur_node,feature=cur_state,reward=r)

    def update_edge(self,cur_node,act,reward,next_node):
        #不能只遵循Q原则更新边
        old_rew = self.Gmemory[cur_node][next_node]['rew']
        old_visits = self.Gmemory[cur_node][next_node]['visits']
        old_q = self.Gmemory[cur_node][next_node]['q']
        candidate_neighs_of_next_node = list(self.Gmemory[next_node])
        if len(candidate_neighs_of_next_node) ==0 :
            # 如果该节点是终点
            tar = reward # R作为目标
        else:
            weight_list = []
            for neighs_i in candidate_neighs_of_next_node:
                weight_list.append(self.Gmemory[next_node][neighs_i]['q'])
            weight_max = np.max(weight_list)
            tar = reward + self.gamma*weight_max
        delta_q = tar -old_q
        new_q = old_q + self.lr* delta_q
        self.Gmemory.add_edge(cur_node,next_node,label=act,rew=reward,visits=old_visits+1,q=new_q)

    def add_pair(self,cur_node,cur_state,act,r,next_node,next_state):
        #一次要加入两个，暂时只使用一个
        #self.Gmemory.add_node(loc,feature=state,reward=r)#如果之前有也直接替换，不考虑做中和吗？
        # 暂时还不能在这里加入，因为node 列表有可能被替代，而这个list不会，所以二者不等长
        #self.StateAttributesDict.append(state)
        # 每次用node_featrue_list从头捋一遍吧
        
        #if [cur_node,next_node] in self.Gmemory.edges():
        if self.Gmemory.has_edge(cur_node,next_node):
            #print("update edge *****************************")
            self.update_edge(cur_node,act,r,next_node)
        else:
            #增加节点和边
            #print("add node")
            if self.check_node_exist(cur_node):
                if self.check_node_exist(next_node):
                    pass
                else:
                    self.Gmemory.add_node(next_node,feature=next_state,reward=r)
            else:
                if self.check_node_exist(next_node):
                    self.Gmemory.add_node(cur_node,feature=cur_state,reward=0)
                else:
                    self.Gmemory.add_node(cur_node,feature=cur_state,reward=0)
                    self.Gmemory.add_node(next_node,feature=next_state,reward=r)
            self.Gmemory.add_edge(cur_node,next_node,label=act,rew=r,visits=1,q=r)

    def node_feature_list(self):
        F = [] #这个矩阵能不能在add pair的时候加入？
        # print("len()",len(self.Gmemory))
        for nodeid in self.Gmemory.nodes():
            # print("i",nodeid)
            # print("feature",self.Gmemory.node[nodeid]['feature'])
            F.append(self.Gmemory.node[nodeid]['feature'])
            
        return F

    def read_edges_of_node(self,read_node):
        # 输入一个节点，得到所有与它相连的动作
        #由于没有找到直接读边的函数，这里先读下一个节点，然后再读边
        temp = self.Gmemory[read_node]
        next_node_candidates = list(temp)
        #print(next_node_candidates)
        action_candidates = []
        value_list = []
        for i in range(len(next_node_candidates)):
            #print("read node",read_node,"next_node ",i,"is ",next_node_candidates[i])
            # if self.Gmemory.has_edge(read_node,next_node_candidates[i]):
            #     print("there is an edge")
            # else:
            #     print("we have no edges here")
            action_candidates.append(self.Gmemory[read_node][next_node_candidates[i]]['label'])
            value_list.append(self.Gmemory[read_node][next_node_candidates[i]]['q'])
        return action_candidates,value_list,next_node_candidates
    
    def seqwriter(self,re_vec):
        # 将一条轨迹写入存储中，这条轨迹是 已经编码 过后的
        # 是否有必要从后往前更新
        for i in range(len(re_vec)):
            self.add_pair(re_vec[i][0],0,re_vec[i][1],re_vec[i][2],re_vec[i][3],0)
