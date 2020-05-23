# GBMRcode
test code for GBMR

1. 构建环境；
2. 编码网络单独训练；
3. 检索方案；
4. 规划方法；
5. 交互方案；
6. 存储结构；
7. 记忆重构；

## 1. 环境构建

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


实现步骤：
1. 先把各个环境安上，看看哪些是目前能用的；
2. 每个环境构建完成之后要有一个基本的调用代码，能让它可视化并完整的运行；
3. 要能有一个基本算法可以在该环境中有一定效果，也就是要求是强化学习本身可以完成的任务

## 2. 编码网络

编码网络的要求：
+ 基本要求是能把视频（图像）输入压缩成向量
+ 如果能在这个过程中完成聚类

候选的方案：
+ tvt 中的编码
+ world model中的编码
+ sorb 中的编码

问题：
+ 一般处理视频类的算法中都会有相应的编码网络，但是他们都不是单独训练的

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

## 规划方案

规划方案的要求：
+ 先完成记忆结构中的规划，也就是离线规划
+ 要能在行进中检索到当前状态及其相似状态，并根据结果形成决策（多点同时规划，自定义深度）

候选方案：
+ Muzero 中的方案

## 交互


## 存储 

存储和编码以及检索息息相关

## 记忆重构

这是重点。

要求：
+ 能够得到状态的聚类
+ 根据聚类结果进行推演

候选方法：
+ 节点特征的聚类
+ 节点特征传播之后聚类
+ 普通的聚类方案

# 整体构造

针对连续问题设计的基本框架，用于调试各个模块的数据流通。在basicframework文件夹中。
测试通过之后，再重新阅读所有参考资料，并部署实验


base 中是基本架构，包含了期望的接口和整体的框架 #2020年5月22日