'''



'''


#这里只提供读写器删除器，那么具体的数据放在什么地方

#记忆读取模块
#参考tvt/memory.py


class MemReader():
    def __init__(self,memory_word_size,num_read_heads,top_k,memory_size,name='MemReader'):
        self._memory_word_size = memory_word_size
        self._num_read_heads = num_read_heads
        self._top_k = top_k
        self._output_size = num_read_heads * memory_word_size

    def read_from_memory(self,read_info):
        # 根据read info 从memory 中查找相应的数据
        # tvt/memory.py中有一个叫read from memory 的函数，可以借用过来
        
        #返回值包含memory_reads, read_weights, read_indices, read_strengths
        #暂时用一个量替代

        mem_reads = read_info
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
        #应该把Gmemory继承过来做内部值，不然每个函数都要用一下
        # 定义一些基础变量和网络
    def _get_location(self,cur_state,Gmemory):

        '''
        先用相似度度量，看看最接近的节点的相似度是否小于某个阈值，如果小于，那么就认为是这个节点，如果不小于，那么就把它当作新节点，
        '''
        if Gmemory.num_nodes_in_Mem() == 0:
            cur_node = 0
        else:
            # 计算 向量 cur_state，和 Gmemory中所有节点特征的相似度 
            print(Gmemory.node_feature_list())
        return cur_node

    def memory_writer(self,Gmemory,epshistory):
        # 写入记忆并进行关联
        #
        length = len(epshistory)
        for i in range(length):
            #写入轨迹中的每一对儿点
            [cur_state,act,r,next_state] = epshistory[i]
            #print(act)
            #先找到写入位置
            cur_node = self._get_location(cur_state,Gmemory)
            print(cur_node)
        Gmemory.testadd(length)
        Gmemory.plotmemory()