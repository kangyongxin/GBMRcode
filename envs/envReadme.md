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
