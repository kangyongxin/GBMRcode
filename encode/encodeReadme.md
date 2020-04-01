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
+ Sorb中并没找到环境编码部分
+ 先从 tvt入手，看看能不能用它的编码和存储检索结构，然后再考虑用world model 做参考
+ 环境方面，先试试pycolab，然后再扩展到Atari上，感觉如果只是编码，两个环境应该都能用

### 整体思路

    2.1 先阅读整个tvt 的结构，梳理清楚（要一气呵成）
    2.2 再把环境安装上
    2.3 测试编码和检索过程是否行的通
    2.4 阅读world model 中环境封装的方法
    2.5 然后用Atari环境测试编码是否行的通

### 2.1 tvt 代码结构

（注意这里是一个batch 一起做的，也就是16个环境一起跑的）

1. 环境构建

        env_builder = pycolab_env.PycolabEnvironment #自己构建的一个环境集合，有两个游戏 key_to_door 和active_visual_match，
        #pycolab_env.PycolabEnvironment是pycolab中env.py 里的一个class  """A simple environment adapter for pycolab games.""" """Construct a `environment.Base` adapter that wraps a pycolab game."""
        #pycolab 本身是一个库，然后这里又用tvt.pycolab.active_visual_match.py key_to_door.py 做了一下包装。key_to_door中直接给出了一个环境的形式

        env_kwargs = {
            'game': FLAGS.pycolab_game,
            'num_apples': FLAGS.pycolab_num_apples,
            'apple_reward': [FLAGS.pycolab_apple_reward_min,
                            FLAGS.pycolab_apple_reward_max],
            'fix_apple_reward_in_episode': FLAGS.pycolab_fix_apple_reward_in_episode,
            'final_reward': FLAGS.pycolab_final_reward,
            'crop': FLAGS.pycolab_crop
        }

        env = batch_env.BatchEnv(batch_size, env_builder, **env_kwargs)#似乎对环境也进行了打包

        ep_length = env.episode_length #没找到在哪儿初始化的
        #在common.py中， 三个数的加和 DEFAULT_MAX_FRAMES_PER_PHASE = {  'explore': 15, 'distractor': 90, 'reward': 15}

2. 智能体定义

        agent = rma.Agent(batch_size=batch_size,
                        num_actions=env.num_actions,
                        observation_shape=env.observation_shape,
                        with_reconstructions=FLAGS.with_reconstruction,
                        gamma=FLAGS.gamma,
                        read_strength_cost=FLAGS.read_strength_cost,
                        read_strength_tolerance=FLAGS.read_strength_tolerance,
                        entropy_cost=FLAGS.entropy_cost,
                        with_memory=FLAGS.with_memory,
                        image_cost_weight=FLAGS.image_cost_weight)

3. 变量定义 

        %   Agent step placeholders and agent step.
        loss, loss_logs = agent.loss(
            observations_ph, rewards_ph, actions_ph, tvt_rewards_ph)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            epsilon=FLAGS.epsilon)
        update_op = optimizer.minimize(loss)
    initial_state = agent.initial_state(batch_size)

4. 初始化

      
        init_ops = (tf.global_variables_initializer(),
                    tf.local_variables_initializer())
        tf.get_default_graph().finalize()

        sess = tf.Session()
        sess.run(init_ops)
        if FLAGS.print_functionname:
            print("after sess.run(init_ops)")
        run = True
        ep_num = 0
        prev_logging_time = time.time()

5. 每个回合内

        for tt in range(ep_length):
            #print("tt in ep_length",tt)
            step_feed = {reward_ph: reward, observation_ph: observation}
            for ph, ar in zip(nest.flatten(state_ph), nest.flatten(agent_state)):
                step_feed[ph] = ar
            
            step_output, agent_state = sess.run(
                (step_outputs, state), feed_dict=step_feed)
            action = step_output.action
            baseline = step_output.baseline
            read_info = step_output.read_info

            # Take step in environment, append results.
            observation, reward = env.step(action)

            observations.append(observation)
            rewards.append(reward)
            actions.append(action)
            baselines.append(baseline)
            if read_info is not None:
                read_infos.append(read_info)

            %第一个操作是相当于普通的RL中得到动作的过程，只是这里顺便得到了state

            %这是在用一个网络根据当前的 观测和奖励 得到 action/baseline/read_info 
            %step_out 和 state 作为输出，现在看看这两个变量的构造
            %这是这两个变量的定义：step_outputs, state = agent.step(reward_ph, observation_ph, state_ph)
            %在rma.py中，agent.step（）有三个输入，两个输出。这是个前向的网络。
                % 三个输入分别是当前的奖励reward， 当前的观测observation，之前的状态（这个状态并不是用observation直接编码得到的状态，而是每走一步都会有一个agent_state,凑一眼它是怎么来的）
                % agent_state = AgentState(core_state=next_core_state ,prev_action=action)  两个东西组成，next_core_state和prev_action, 这两个是self._core 的输出得到的，self._core的输入是（features , 和 prev_state.core_state）, 
                    其中features是self._encode 把观测奖励之前的动作编码得到的。再往前追溯就是各个分量各自的编码网络了，这个网络应该有一定的通用性(****)。
                     而prev_state.core_state 是一个一直都在更新的变量，也就是这个agentstate的输出就作为下个输入的prev_state.(这个结构是RNN循环的基础，并不是做图当中必须用到的)
                %知道了两个输入之后，回头来看self._core中对特征和上一个状态做了什么？
                %在rma.py中的_RMACore()类中（RMA RNN Core）。但是这个class并没有显示的输出，可是在agent中调用的时候却有两个输出，怎么得到的呢？因为自动调用了一个_build 函数，这个函数有两个输出，一个是 core_outputs,一个是h_next。 
                    %core_output是根据policy 网络算出来的，这个网络的输入包含了两部分：一部分是（特征features, 之前的状态 h_prev.controller_outputs， 记忆中的之前状态h_prev.mem_reads）然后再编码一下。另一部分是policy_extra_input与记忆相关。（****这里会有如何存取记忆的方法）具体的在memory.py中
                    %h_next 是拿RNN算出来的，同样是与记忆和当前的mem_reads有关
                %step()的另一个输出是step_output()[个人感觉这个step是RNN的step,放在这里容易和RL 的step混了]
                    %它有action baseline read_info三部分，也是根据core_output 算出来的值。这里的baseline是干啥的呢？在_core中构造coreoutputs的时候，从policyoutputs中继承来的

            %第二个操作是在得到动作之后，在环境中执行动作，observation, reward = env.step(action)，

            %第三步把这些东西记录下来，然后用loss 训练网络（如果不加入tvt reward的话）

    如果不加入tvt reward，直接用这一步也是可以的，包含了编码网络，记忆的存读都在agent.step中完成了。 

    这是个强化学习过程，没有训练任何网络。训练网络是在回合结束之后。   


6. 每个回合结束后



            # Stack the lists of length ep_length so that each array (or each element
            %# of nest stucture for read_infos) has shape (ep_length, batch_size, ...). 这里的read infos 是有一定的格式的（要关注******）
            observations = np.stack(observations)
            rewards = np.array(rewards)
            actions = np.array(actions)
            baselines = np.array(baselines)#没搞清楚为啥要用这个baseline
            read_infos = nest_utils.nest_stack(read_infos)#（有一个叫nest_utils.py的文件Equivalent to np.stack, but works on list-of-nests）
            print("observations",observation.shape)
            print("baselines",baselines.shape)
            # Compute TVT rewards.
            if FLAGS.do_tvt:
                tvt_rewards = tvt_module.compute_tvt_rewards(read_infos,
                                                        baselines,
                                                        gamma=FLAGS.gamma)
            else:
                tvt_rewards = np.squeeze(np.zeros_like(baselines))

            # Run update op.
            loss_feed = {observations_ph: observations,
                        rewards_ph: rewards,
                        actions_ph: actions,
                        tvt_rewards_ph: tvt_rewards}
            ep_loss, _, ep_loss_logs = sess.run([loss, update_op, loss_logs],
                                                feed_dict=loss_feed)

有三个主要操作，第一个是根据这一个回合的数据进行转换，构造读取的表头信息read_info 和 baseline； 第二个操作是把相应的信息输入tvt_rewards.py 中 Compute TVT rewards from EpisodeOutputs，重点在于是根据回合信息计算所有的状态上得到的额外奖励，返回 Returns: An array of TVT rewards with shape (ep_length,)； 第三个操作是把当前的到的这些数据构造loss 并进行训练。

接下来，先关注如何计算tvt reward，再关注如何训练
+ tvt reward 的计算： 输入是readinfo,baselines, 二者的维度都是ep_length* bacth_size。 这些东西在进行交互的同时就已经形成，并存储在memory中了。读的时候，先给他解包，然后_compute_tvt_rewards_from_read_info（）【源代码中采用异步提交任务的方法submit(fn, *args, **kwargs)异步提交任务，调用相应函数】 在这个函数中，先把batch分开了，每个batch是独立操作的，调用 _tvt_rewards_single_head（）【Compute TVT rewards for a single read head, no batch dimension.返回一串tvt rewad  Returns:An array of TVT rewards with shape (ep_length,).】这个函数里是一个寻找相关性，并进行叠加的过程。仔细看看和记忆有什么关系？它和记忆存储并没有直接关系
    + 它只是根据当前的轨迹中见过的进行比较，那要外部存储做什么？
    + 先搞清楚用谁检索谁，其实每个样本都对应一个baseline，然后我们检索的时候根据掩码，然后读出一个数来，
+ loss 的构建（loss, update_op, loss_logs）
    + loss 在 rma.py中定义：total_loss = a2c_loss + recon_loss + read_reg_loss
    + a2c_loss = trfl.sequence_advantage_actor_critic_loss(这是个DeepMind建立在tensorflow之上的一个强化学习库)
    + recon_loss = losses.reconstruction_losses()（这是我们需要的****image_loss）
    + read_reg_loss = losses.read_regularization_loss()也是一个专门的函数"""Computes the sum of read strength and read key regularization losses."""（用来算正则化损失）

绘制整个蓝图，看看剩下的任务还差什么？

testmain.py 负责整体调度 在encode 文件夹中：
pycolab文件夹用来包装环境
rma.py 用来定义智能体，包括编解码网络，loss构建函数的调用，
reconstruct.py (之前没有)我们要用它做推演，取代rma.py中的step模块（它名字起的不好）
loss.py用来定义不同的loss, 
memory.py 用来构造记忆，读和写

testmain.py 负责整体调度 
+ 在encode 文件夹中  python testencode.py
+ 库函数，absl-py（做分布式调度） 未安装，six(解决py2 py3兼容的问题)
+ tensorflow.contrib 用nest  对结构像素进行包装，但是这个东西只有tensorflow 1.14才有。
+ batch env 中原先是整体封装的，现在我们尝试只做一个
+ 怎么显示环境当前状态呢，render()不好用，可不可以从humanui中找到相应的封装？(没搞清楚)

+ 随机动作 ，现在只能读出来个数，没法得到整个动作空间，但是它是从零开始的，【0，1，2，3】 定义一个agent 给出随机策略

angent 定义,由于构造网络的时候用到sonnet 所以现在要pip sonnet ,他同时要求tensorflow probability 

+ 构建编解码网络，这里要想办法解决sonnet的问题，先把网络结构复现出来。
+ 在源代码agent的参数中image_code_size=500和observation_shape 并不是同一个值。500 dimensions是论文中最后给出的编码长度。文章中的网络结构是：
    + 输入：64*64*3 的tensor
    + 6层resnet，分段线性激活函数（rectified linear function），64 个通道输出，The strides for the 6 blocks are （2; 1; 2; 1; 2; 1），输出图片是八倍下采样，8 * 8 *64 的输出
    + flatten 成 4096
    + 做全连接 得到500 维的编码
+ 对比来看，这个比源码中的输入比较小，是5*5*3 （key to door）,但输出维度依然被做成了500
+ 现在面临两个问题，一个是编解码网络的复现，两外一个是搞清楚后来的重构误差如何计算，也就是网络如何训练？
+ 尝试直接拿tensorflow 重现snt的网络结构,或者把tensorflow 弄到2.1 ，然后看看能不能把tensor1.14的版本改到tensorflow2.1上

+ 目前看到的就一个在tenv.py中，colours = nest.map_structure(lambda c: float(c) * 255 / 1000,self._game.colours)
+ 在1中的引用格式
tf.contrib.framework.nest.map_structure

tf.nest.map_structure(
    func,
    *structure,
    **kwargs
)
+ 在2中的引用格式
tf.nest.map_structure(
    func, *structure, **kwargs
)

+ 又改了很多tf2.0和新版sonnet用到的函数

+ 现在出现的问题是Shape (5, 5, 3) must have rank 4。说明我们的输入维度不够，NHWC[NHWC：[batch, in_height, in_width, in_channels]]

+ Expected floating point type, got <dtype: 'uint8'> 重读代码肯定有在哪里进行数据格式转换的
x = tf.constant([1.8, 2.2], dtype=tf.float32)
tf.dtypes.cast(x, tf.int32)  # [1, 2], dtype=tf.int32

+ 一个重要的问题，就是这些变量不能直接用，要通过各种通道传进去,所以还得从头来, 开始在主函数中构建loss,并初始化网络，尝试训练完成之后做存储。
+ 发现tf2.0已经抛弃了placeholder 这种东西了。简单试一下keras构建网络，用原先的sonnet 太麻烦
+ keras 构建网络的时候一般由一个init来定义要使用的层，由一个call来定义层之间的连接。
+ vae ,这里的vae编码部分只是一层投影，然后外挂两个输出，一个是z_mean,一个是z_log_var,然后从中采样一个z,作为编码；解码部分，一个投影层，一个输出层。
+ 然后用一个叫VAE的class 把他俩连接起来，输入是 input,中间层是z,输出是reconstructed， 构建loss 也在这里
+ 注意这里的vae继承了很多tf.keras.model，所以会有很多自带的功能比如 compile,fit,等等

+ 我们要做到是，对一组图像进行编码，然后重构，并能够把中间结果拿出来用,这里的dense 输出是32维的。

+ 接下来就是用同样的结构复现Sonnet 构造的编解码器。

+ 优化器是在主函数里定义的，编码器的结构在智能体内部定义
+ 先解析原有的网络结构：
    + encode， 输入是图像，进行卷积，然后铺平，得到编码
    + decode，输入是编码，进行一步全连接，然后反卷积，得到同等维度的图像
+ 输入是把图像拉平了，所以要调整输入维度。
+ 先把函数融合进去，再想着调整输入
+ 现在要把obs做成和x_train相似的结构，把agent中的输入调整成与 obs相同的长度。

遗留问题，编码的长度是32位，能表征很多信息，这里的格子是25个，我们其实用5位0，1编码就能表征，这里暂时不去细究，先把整个网络结构。

接下来考虑，存储和索引。

