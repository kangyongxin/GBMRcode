
'''
计算相似度的函数
输入可以是一个 元素和 一个元素list
此时得到的是一个向量，表示相似度
也可以是两个元素list
那么输出就是一个矩阵，

'''
import tensorflow as tf

_EPSILON = 1e-6

def _vector_norms(m):
  squared_norms = tf.reduce_sum(m * m, axis=1, keepdims=True)
  return tf.sqrt(squared_norms + _EPSILON)

class CosineSimilarity():
    """Cosine-Similarity.
    copy from dnc.addressing.py
    Calculates the cosine similarity between a query and each word in memory, then
    applies a weighted softmax to return a sharp distribution.
    """
    def __init__(self,num_heads,word_size,name='cosine_similarity'):
        self._num_heads = num_heads
        self._word_size = word_size
        
    def onevsall(self, keys, memory):
        """Connects the CosineWeights module into the graph.

        Args:
        memory: A list of 3-D tensor of shape `[ 1, word_size]`.
        keys: A 3-D tensor of shape `[ 1, word_size]`.
        

        Returns:
        Weights tensor of shape `[batch_size, num_heads, memory_size]`.
        """
        # Calculates the inner product between the query vector and words in memory.
        similarity = []
        for i in range(len(memory)):
            # print("keys",keys)
            # print("memory[i]",memory[i])
            dot = tf.matmul(keys, memory[i], adjoint_b=True)

            # Outer product to compute denominator (euclidean norm of query and memory).
            memory_norms = _vector_norms(memory[i])
            key_norms = _vector_norms(keys)
            norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)

            # Calculates cosine similarity between the query vector and words in memory.
            similarity.append(dot / (norm + _EPSILON))
        return similarity
