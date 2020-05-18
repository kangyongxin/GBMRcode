


'''

根据当前的抽象图，重新构造连接的问题

'''
import  networkx as nx 

class MemReconstructor():
    def __init__(self,memory_word_size,name='MemReconstructor'):
        self.memory_word_size = memory_word_size

    def reconstruct_by_abstract_graph(self,abstract_graph,external_graph):
        print("read in abstract graph :", abstract_graph.num_nodes_in_Mem())
        print("replay external graph :",external_graph.num_nodes_in_Mem())

        for nodei in abstract_graph.Gmemory.nodes():
            for nodej in abstract_graph.Gmemory.nodes():
                if nodei == nodej:
                    continue
                if nx.algorithms.shortest_paths.generic.has_path(external_graph.Gmemory,nodei,nodej):#center_list[i],center_list[j]):
                    #print("reconstruct")
                    path = nx.shortest_path(external_graph.Gmemory,nodei,nodej)
                    #print("path",path)
                    temp = []
                    for idx in range(len(path)-1):
                        pair_start = path[idx]
                        pair_end = path[idx+1]
                        # print(pair_start)
                        # print(pair_end)
                        # print(external_graph.Gmemory[pair_start][pair_end]['label'])
                        # print(external_graph.Gmemory[pair_start][pair_end]['rew'])
                        temp.append([pair_start,external_graph.Gmemory[pair_start][pair_end]['label'],external_graph.Gmemory[pair_start][pair_end]['rew'],pair_end])
                    #self.MemoryWriter(temp)更新现有边上的权重
                    #这里因为已知节点序号，所以可以直接写进内存
                    external_graph.seqwriter(temp)
                else:
                    #print("no path for reconstruct")
                    pass
        return True


