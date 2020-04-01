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

 6. 例子在openai中有最简单版本,从官网上找到的例子比较靠谱，testenvs()中定义了一个随机的探索方案，这里可以看到有视频帧作为观测的游戏可以用，那么接下来看看两个算法是如何处理和包装这个环境的。
    
    + baselines 中common 文件夹中的cmd_util.py包装参数，wrappers.py包装环境, 这里好像默认使用的是mpi,但是不知道具体有什么用途。可以用 make_atari包装，也可以再包装一层wrap_deepmind 都在atari_wrappers.py中。
    + 环境要用
        python testforAtari.py--env=MsPacmanNoFrameskip-v0
    + 仍然没有搞懂多进程如何封装环境

7. 接下来试试t2t中的环境如何封装
    + 基本流程：从trainer_model_based.py中开始,环境设置在rl_utils.setup_env（），然后再每个epoch中先train_world_model()，然后train_agent()环境是world model， 之后train_agent_real_env(),最后再env.generate_data()
        + setup_env() 中主要函数是T2TGymEnv(): from tensor2tensor.data_generators.gym_env import T2TGymEnv 重新封装了一个环境，这个环境的测试在gym_env_test.py中
        + train_world_model()中以train_supervised（）为主，再下一级是trainer_lib.create_experiment_fn和 trainer_lib.create_run_config
        + 同样trainer lib 中也有相应的test文件，其实这个习惯很好，每写完一个就写一个test.py 这里似乎是在构造一个分布式训练的平台
        + train agent() 是一个在虚拟环境中训练智能体的网络
        + train agent real env() 是在实际环境中训练，
        + 两个训练训练用的是同一函数train()，参数simulated 不同，一个True 一个False. 这个训练器在rl.dopamine_connector.py中。y有空看看这个train是怎么写出来的。

## vizdoom

SPTM中有相关代码，但是环境是要通过setup.sh 调用conda_env.txt 进行安装。

R-network is called 'edge' and L-network is called 'action' throughout the code

train_action_predictor.py 和train_edge_predictor.py 是两个训练函数，但是没搞懂怎么训练出来的。

## SoRB中的环境

main.py中分了以下几个部分：
1. 初始化： 
+ tf.reset_default_graph()
+ tf_env = env_load_fn() 这里默认加载的是一个叫FourRooms的环境
+ eval_tf_env = env_load_fn（）看这个架势，用来测评和用来训练的环境不同
+ agent = UvfAgent()

2. 可视化环境：
+ 推演过程的可视化

3. 填充replay buffer
+ 用到的是eval_tf_env

4. Compute the pairwise distances

5. Graph Construction 构建图
+ 重点关注节点是怎么来的

6. Ensemble of Critics

7. Search Policy
+ SearchPolicy（）

8. search path 


9. rollout with search 用得到的策略进行推演

接下来从初始化的环境看起，重点要知道给智能体的状态输入到底是什么？

1. 初始化
1.1 tf.reset_default_graph 与图无关，只是设置默认的tf计算结构
1.2 tf_env = env_load_fn() 这个env好像是自己写的，env 的名字是直接给出的“fourROOM”,最大步骤数默认是20 ，terminate_on_timeout=False 
+ gym_env = PointEnv(wall, resize)wall是可选的，可以先试一个。在模块引入中有一个tf_agent 是什么？
+ wall 是一堆[0,1]数组，按名称索引， 动作空间和状态空间都是用gym.spaces.box()构造的，在——compute_apsp中构图。（这个构图的方案值得借鉴）包含了采样状态（sample_empty_state）计算最短距离（getdistance）更新状态 （step）, 动作的函数设计为action = np.clip(action, self.action_space.low, self.action_space.high)，step中设计了更新细节

+ tf_agents暂时没装好，先用一下别的,环境是可用的。但是源代码想用tf.agent 把这个环境封装成gym格式的，状态就是一个表示位置的数组

+ 看看实际应用过程中是怎么从环境中获得状态的，然后对比一下论文中的实验结果


1.3 eval_tf_env() 这里唯一的区别是terminate_on_timeout=True 原文注释是Whether to set done = True when the max episode steps is reached.表现在函数中就是用wrappers.TimeLimit还是用 NonTerminatingTimeLimit 来包装环境的问题。

1.4 agent= UvAgent() 

2. 根据当前策略推演
get_rollout(),得到起止点和中间途径点。


1.n search_policy中构建了两幅图，一幅是原先的g,边权是各个节点之间的距离distance，另一幅是g2,是根据起止点的距离远近来算。


## tvt 的环境

这里的开篇是把16个环境一起打包训练的，是不是可以先摘出来。

rma.py 是结构
main.py 是流程


强化学习的网络训练，通常都是用奖励做监督信号，一般都是在运行一个回合或者一个step之后，得到的观测拿过来训练网络。在tvt中，构造标签的时候直接将tvt_rewards加入loss中。

tensorflow 是先构造网络和loss结构，再把数据喂进去。
这个代码集成了很多已有的东西，比如sonnet,trfl,contrib, 这些都是人家的积累和底气，我们要逐渐把它熟悉，争取能够使用
1. 初始化 env_builder; env ; agent = rma.Agent;
目标是与我们自己的图结构相结合。

在tvt.pycolab.env.py中对最原始的pycolab环境进行了包装，包括对观测的RGB转换，reset(),step(),动作状态空间等等，这些如果我们想从头做起都是必须的。

也就是说无论用哪个环境，我们不能绕过的是要对环境进行包装和状态的转换，

tvt的编码部分是和后面一起训练的，也就只有它有编码存储的全套工作，所以兜兜转转一圈又回到了起点，但是我们积累了很多环境包装和应用的手段，所以接下来的任务是，把这个tvt的环境包装和编码存储部分搞清楚，并尝试运行。

这个代码能完成两件事情就行，1能够编码，并训练编码器；2.能够记忆，并提供检索方案

现在先不考虑强化学习任务，只是得到一个矩阵，进行编码，然后从记忆中得到最相似的。