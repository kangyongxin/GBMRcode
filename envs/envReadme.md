# 环境

我们对环境的要求：
+ 要是视频输入；
+ 要有多个奖励；
+ 适应于探索任务；

候选环境：
+ Gridmaze
+ 动态的给gridmaze
+ atari
+ vizdoom
+ pycolab
+ tvt 的环境
+ sorb 的环境
+ sptm 的环境

每个环境试运行和安装

## Grid maze

\# python ./envs/maze/maze3_env20.py

## Atari


 1. atari 环境在baseline中并没有直接使用，而是经过了层层包装才能适应网络训练，我们不准备重新包装，但是要知道怎么使用。

 注： https://blog.csdn.net/qq_41832757/article/details/104390909 中有关于如何包装环境的atari_wrappers.py的解析。

 先试试直接使用是否可以。

 2. Atari 和gym中会有不同的环境类型

 3. 可能出现的一个问题是，如果在baseline的基础上做，那么就要以tensorflow为基础进行，但是后面用到的图聚类可能又会用到pytorch 不知道二者是否兼容。

 4. world model 是否是在baseline 的基础上完成的？world model 中对环境的包装同样是基于baseline中的，只不过是换了个名字 gym_utils.py 所以我们重点不是去构造那个函数，而是使用这个包装。

 5. 现在的任务是尝试根据这两个（baseline 和 world model ）的应用，总结Atari环境的使用方法。先把一个随机动作的测试文件跑起来。据说没有可视化的环境包装。