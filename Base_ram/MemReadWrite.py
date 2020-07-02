'''
读写器模块
读，根据readinfo 在外部存储器中读取相应记忆
删，根据即将写入的内容删除相似的内容，或者删除使用频次过高的记忆
写，将轨迹写入外部存储


#记忆读取模块
#参考tvt/memory.py
# 相似度度量参考dnc中access.py的调用逻辑
# 删除部分也准备使用dnc 相关代码
'''
import utils

class MemReader():
    def __init__(self,memory_word_size,num_read_heads,top_k,memory_size,name='MemReader'):
        self._memory_word_size = memory_word_size
        self._num_read_heads = num_read_heads
        self._top_k = top_k
        self._output_size = num_read_heads * memory_word_size

    def read_from_memory(self,read_info,Gmemory):
        # 根据read info 从memory 中查找相应的数据
        # tvt/memory.py中有一个叫read from memory 的函数，可以借用过来   
        # 返回值包含memory_reads, read_weights, read_indices, read_strengths     
        read_node = read_info
        #找到这个节点所有的边，要找到一个聚类中心点所对应的这一类中所有的节点，也就是在外部存储图中所有n阶邻域中的点
        #找到这个邻域中所有点的可能决策
        #问题出现在，邻域存在什么地方，我们在build abstract graph的过程中存了一个pairs
        acts,values,next_nodes = Gmemory.read_edges_of_node(read_node)
        mem_reads = acts
        return mem_reads

class MemErase():
    def __init__(self, memory_word_size,memory_size,name='MemErase'):
        self._memory_word_size = memory_word_size
        # 定义一些基础变量和网络

    def memory_eraser(self,Gmemory,location):
        # 我们要知道删除什么样的记忆，并且删除相关连接
        #
        
        Gmemory.testminus(location)
        Gmemory.plotmemory()

        # 不需要return 直接更改原图

class MemWriter():
    def __init__(self, memory_word_size,memory_size,name='MemWriter'):
        self._memory_word_size = memory_word_size
        self._write_content_similarity = utils.CosineSimilarity(1,memory_word_size,name="write_content_similarity")
        #应该把Gmemory继承过来做内部值，不然每个函数都要用一下
        self._TH =0.999 #计算阈值，如果大于这个值新建节点(余弦距离的时候有正有负，所以暂时取正的)
    
    def _get_location(self,cur_state,next_state,Gmemory):

        '''
        先用相似度度量，看看最接近的节点的相似度是否小于某个阈值，如果小于，那么就认为是这个节点，如果不小于，那么就把它当作新节点，
        '''
        '''
        这个函数后续要根据读写索引进行更改
        '''

        if Gmemory.num_nodes_in_Mem() == 0:
            print("an empty memroy")
            cur_node = 0
            next_node =1
            
        else:
            # 计算 向量 state，和 Gmemory中所有节点特征的相似度 
            #print("memory has ",Gmemory.num_nodes_in_Mem(),"nodes")
            cur_similarity = self._write_content_similarity.onevsall(cur_state,Gmemory.node_feature_list())
            next_similarity = self._write_content_similarity.onevsall(next_state,Gmemory.node_feature_list())

            #print("similarity",similarity)
            if max(cur_similarity)<self._TH:
                if max(next_similarity)<self._TH:
                    #print("add new node",max(similarity))
                    cur_node = Gmemory.num_nodes_in_Mem()
                    next_node = Gmemory.num_nodes_in_Mem()+1
                    
                else:
                    cur_node = Gmemory.num_nodes_in_Mem()
                    next_node = next_similarity.index(max(next_similarity))
            else:
                if max(next_similarity)<self._TH:
                    cur_node = cur_similarity.index(max(cur_similarity))
                    next_node = Gmemory.num_nodes_in_Mem()
                else:
                    cur_node = cur_similarity.index(max(cur_similarity))
                    next_node = next_similarity.index(max(next_similarity))
            
        return cur_node,next_node#,new_Flag
    
    def memory_writer(self,Gmemory,epshistory):
        # 写入记忆并进行关联
        #
        length = len(epshistory)
        for i in range(length):
            #写入轨迹中的每一对儿点
            [cur_state,act,r,next_state] = epshistory[i]
            #先找到写入位置
            cur_node,next_node = self._get_location(cur_state,next_state,Gmemory)
            if cur_node == next_node:
                Gmemory.add_single_node(cur_node,cur_state,act,r)
            else:
                Gmemory.add_pair(cur_node,cur_state,act,r,next_node,next_state)
        Gmemory.testadd(length)
        Gmemory.plotmemory()