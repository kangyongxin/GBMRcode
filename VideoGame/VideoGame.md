# 目标

在视频游戏中得到可用效果（包括能够得到一定的奖励，同时能够通过abstract得到相应的图表示）

# 步骤

1. 现在有两个基本框架，稍作整合，要求是只要简单切换给定的环境名称就能得到不同环境中的结果
    + 主要难点出现在环境观测到状态的转换上
    + 然后是对aggregate参数的训练要准备相同的数据格式

2. 要对中间过程可视化，明确知道算法运行的每一步都做了啥
    + 这个可视化的结果不仅对调参有用，对最后的结果说明也是有用的

3. 批量调试，适应不同的环境，看看什么样的环境更容易取得好的效果

4. 参数训练，让Graphsage完整地运行起来


# 笔记

1. 整合代码

用到的参考：
DetailShown 中的runGNN_Detail.py 及 readme.md .
Base_ram 中的run_MsPacman_ram.py 及 readme.md .

+ 环境构造问题：maze是一个自己构造的环境，需要单独的文件夹； Atari直接调用就行； 仍然尝试用flag做超参数读入
+ CUDA_VISIBLE_DEVICES=1 python run_videogame.py --env_name='maze'
+ CUDA_VISIBLE_DEVICES=1 python run_videogame.py --env_name='MsPacman-ram-v4'
+ obs2stat
+ 接下来要搞清楚构图的基本要素之间的关系，用随机动作得到的轨迹构图并进行显示，我们要给节点设计什么样的特征更有利于后面的传播同时方便可视化
    + 显示的时候显示的label 是什么？是那个节点的名字，我们在用一个label自己打上去的，在大的环境中，我们不能再自己打标签了，就按照先来后到算吧
    + 由于有abstract memory 和 external memory 二者可能有相似结构，所以就给他们一个外挂一个存储class


    + 把read write 的功能集合到memory 中，其中有一个相似度比较来确定是哪个点的环节，暂时还不能省掉，而且是后续的一个重要的拓展点
    + 先把轨迹写进图中，然后再尝试可视化出来


+ 主要区别是pooling 换成了聚类