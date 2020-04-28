# 定义外部存储
# 核心是一个有向图
# 这里要做到可读写

import networkx as nx

class ExternalMemory:
    def __init__(self, memory_size=100):
        self.Gmemory=nx.DiGraph()
        self.StateAttributesDict={}
        self.memory_size = memory_size
        self._test_n =50
        self.add_node_for_test()

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

    def add_node_for_test(self):
        self.Gmemory.add_node(0,feature=[1,2,3],value=12.3)
        self.Gmemory.add_node(1,feature=[23,33,11],value=43.3)
        self.Gmemory.add_node(2,feature=[23,33,11],value=43.3)

    def node_feature_list(self):
        F = []
        for i in range(len(self.Gmemory)):
            F.append(self.Gmemory.node[i]['feature'])
        return F

    
    