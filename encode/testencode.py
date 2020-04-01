# 尝试做编码存储
# 所有阶段任务都以t开头命名文件（库函数调用除外）
'''
1. 先把环境pycolab 中的keybord拿出来试试，仿照pycolab文件夹用来包装环境
2. 整体运行随机探索过程，然后对观测动作有所了解，构建testmain.py
3. 引用编码网络对观测observation进行编码，解码，仿照TVT中rma.py构造重构loss
4. 仿照memory.py把探索过程中的编码结果state存在外部存储，并进行检索，比较自己检索结果是否精确

'''

#环境构建过程中用到的库
import numpy as np 
from six.moves import range
from six.moves import zip
from absl import app
from absl import flags
from tpycolab import tenv as pycolab_env
import encodeAI
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
#参数定义
FLAGS = flags.FLAGS
flags.DEFINE_enum('pycolab_game', 'key_to_door',
                  ['key_to_door', 'active_visual_match'],
                  'The name of the game in pycolab environment')
# Pycolab-specific flags:
flags.DEFINE_integer('pycolab_num_apples', 10,
                     'Number of apples to sample from the distractor grid.')
flags.DEFINE_float('pycolab_apple_reward_min', 1.,
                   'A reward range [min, max) to uniformly sample from.')
flags.DEFINE_float('pycolab_apple_reward_max', 10.,
                   'A reward range [min, max) to uniformly sample from.')
flags.DEFINE_boolean('pycolab_fix_apple_reward_in_episode', True,
                     'Fix the sampled apple reward within an episode.')
flags.DEFINE_float('pycolab_final_reward', 10.,
                   'Reward obtained at the last phase.')
flags.DEFINE_boolean('pycolab_crop', True,
                     'Whether to crop observations or not.')


flags.DEFINE_boolean('print_functionname', True,
                     'Whether to print_functionname.')



def tmain(_):
    print("hello world!")
    # 调用环境
    batch_size = 1
    env_builder = pycolab_env.PycolabEnvironment
    env_kwargs = {
      'game': FLAGS.pycolab_game,
      'num_apples': FLAGS.pycolab_num_apples,
      'apple_reward': [FLAGS.pycolab_apple_reward_min,
                       FLAGS.pycolab_apple_reward_max],
      'fix_apple_reward_in_episode': FLAGS.pycolab_fix_apple_reward_in_episode,
      'final_reward': FLAGS.pycolab_final_reward,
      'crop': FLAGS.pycolab_crop
      }
    # 封装环境(先不封装，只跑一个)
    print(env_kwargs['game'],env_kwargs['apple_reward'])
    env=env_builder(**env_kwargs)
    ep_length = env.episode_length# 在key_to_door的环境中定义的
    print(ep_length)

    #定义智能体
    agent = encodeAI.Agent(num_actions=env.num_actions,obs_size=75)

    #编码器
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    agent._vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

    # # 定义网络相关的ph
    # batch_shape = (batch_size,)
    # observation_ph = tf.placeholder(
    #   dtype=tf.uint8, shape=batch_shape + env.observation_shape, name='obs') 
    # state = agent.step(observation_ph)
    # #其实一开始用的就是unit8
    # # Update op placeholders and update op.
    # observations_ph = tf.placeholder(
    #   dtype=tf.uint8, shape=(ep_length + 1, batch_size) + env.observation_shape,
    #   name='observations')

    
    # Do init
    # init_ops = (tf.global_variables_initializer(),
    #             tf.local_variables_initializer())
    # tf.get_default_graph().finalize()

    # sess = tf.Session()
    # sess.run(init_ops)
    #测试编码网络
    # (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    # print("xtrain.shape",x_train.shape) 
    # #28*28=784
    # #print("xtrain",x_train[0])
    # # plt.imshow(x_train[0]) # 显示黑白图像
    # # plt.savefig("x_train0.png")
    # # plt.imshow(x_train[1])
    # # plt.savefig("x_train1.png")
    # x_train = x_train.reshape(60000, 784).astype('float32') / 255
    # # vae = VAE(784,32,64)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # agent._vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    # agent._vae.fit(x_train, x_train, epochs=1, batch_size=64)
    # x_test0 = x_train[0:2]
    # print(x_test0)
    # state1 = agent._im2state(x_test0)
    # # state2 = im2state(x_test1)
    # print(state1)


    # 执行循环
    
    while True:
        observation, reward = env.reset()
        observations =[observation]
        #print(observation.shape())
        #env.trender(observation)
        for tt in range(ep_length):
            #env.trender()#没有一个直观的显示
            obs = observation.reshape(1,75).astype('float32') / 255
            state = agent._im2state(obs)
            print(state.shape)#（1，32）
            # take action according to state
            action= agent.TakeRandomAction()
            #print("action",action)
            observation_, reward = env.step(action)
            #print("observation",env.observation_shape)
            #print("reward:",reward)
            observation= observation_
            observations.append(observation)# 因为后面要以batch的形式输入，要把这个三维度数组变成四维的
        
        
        # 编码器训练，每个回合训练一次，也可以选择叠加的形式
        observations = np.stack(observations)
        print("obs.shape",observations.shape)
        input_train = observations.reshape(ep_length+1, observations.shape[1]*observations.shape[2]*observations.shape[3]).astype('float32') / 255
        print("imput.shape",input_train.shape)
        agent._vae.fit(input_train, input_train, epochs=1, batch_size=64)
        # tf.dtypes.cast(observations,tf.float32)
        # states = agent.obs2state(observations)
if __name__ == '__main__':
   app.run(tmain) 