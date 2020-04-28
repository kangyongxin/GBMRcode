# 记忆的存储和检索

先不考虑记忆关联的问题，仅仅考虑把记忆存起来，然后根据相似度进行读取。

至于如何根据图结构检索到更多的记忆，或者做推理，是下一步要完成的任务。

参考DNC的存储检索方案（tvt里也有），环境和编解码还用之前的（重新整理一遍）

## 3. 检索方案

检索方案要求：

+ 给定当前的状态（或其编码），要能够找到相应的记忆单元（或者相似的记忆事件）

候选方案：

+ 直接进行对比
+ 直接用函数（网络）映射
+ tvt 中检索结构
+ dnc的检索方法

实时步骤：
+ 先试试tvt中是如何检索的
+ 然后再考虑图中的检索是不是能完成


## 3.1 tvt中的检索方案

main.py中每一步都根据当前状态得到一个read_info,这个read_info 在后来重构的过程中也用到了。read_info 是通过agent.step（obs, reward）得到的。这里agent.step 可能就是智能体自己想一下，可以把我们的推理写进去，我们用state 做输入，改名为agent.infer(state,reward)，输出为一个动作决策。tvt中是把存和取都写到这个agent.step中来了。我们也暂且先写到一起，主要关注是怎么进行存储和检索的，至于检索哪些内容要根据图结构来做。

+ 这里格外加了一个memory.py的文件夹，包含了memorywriter, memoryreader 等函数

+ 但是还是用sonnet 构建的，LSTM是在RMAcore中的，所以我们的图也是加在那个位置的，external 是它的一个辅助，记忆的存储和检索并不具有LSTM的特性，应该可以放心使用。

+ 源代码中有一个重新编码的过程，这里暂时省略。

+ 源代码中是用LSTM作为控制器，来调用外部存储
+ 而我们要用Graph作为控制器，来调用外部存储

+ 这里的LSTM是训练出来的，那我们的GNN如何训练，是一个值得思考的问题，另外，这个训练的好处在那儿呢？

+ feature ， pre_state是输入，pre_state是LSTM中的状态，包含了该轨迹之前的信息，在主函数中叫agent_state.

+ 把输入输出搞清楚，并搞清楚训练的过程是什么？会给我们带来什么优势.

+ 这里的memoryreader 有两个输入，两个输出，我们分别看一下对应什么?还有，为啥先读，后写？
    + 输入1 controller out， 这是调用者控制器的输出，看看到函数内部它们是什么用处，对于memoryreader类，输入是两个，一个对应的是read_inputs,就是待查询的量，一个是mem_state, 之前的记忆。都是以batch的形式输入的，第一个维度是batch.
    read_inputs （batch 个）进来之后要经过一个生成函数，生成相应的索引, 
    + _keys_and_read_strengths_generator 线性待训练函数，输入是 ([B,...])先用flatten展平，然后送进线性层，参数是要训练的，输出是output 维度（ output_dim = (memory_word_size + 1) * num_read_heads ）每一行的长度加一是什么意思？ 然后乘以读取的条数。
    + 前h列（h是要读取的个数）是flat_key, 后h列是权重。
    + 几个变量的维度：
    read_key 是b,h,m
    mem_state 也是三维的，他俩只有第二个维度不同
    read_strenths 是两维的，b h
    那么问题是read_strength和read_weight 什么关系呢？

    + 输入2 h_prev.memory，这里是之前的记忆，这个记忆是什么时候生成的呢？

    + 输出1 mem_read

    + 输出2 read_info



