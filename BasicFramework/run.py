'''
康永欣 20200420 重新规划，希望能够用最简模块，实现数据流的通畅



'''
import numpy as np 
from six.moves import range
from six.moves import zip
from absl import app
from absl import flags
from envs.tpycolab import tenv as pycolab_env
import GBMRagent
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
'''
参数定义,定义需要后期可能需要批量调节的超参数
'''

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
flags.DEFINE_integer('memory_size', 1000,'the number of nodes we are able to store in the graph.')


def main(_):
    print("Hello world! ")
    # 环境构建
    env_kwargs = {
      'game': FLAGS.pycolab_game,
      'num_apples': FLAGS.pycolab_num_apples,
      'apple_reward': [FLAGS.pycolab_apple_reward_min,
                       FLAGS.pycolab_apple_reward_max],
      'fix_apple_reward_in_episode': FLAGS.pycolab_fix_apple_reward_in_episode,
      'final_reward': FLAGS.pycolab_final_reward,
      'crop': FLAGS.pycolab_crop
      }
    print("env_kwargs: ",env_kwargs)
    env_builder = pycolab_env.PycolabEnvironment
    env=env_builder(**env_kwargs)
    ep_length = env.episode_length# 在key_to_door的环境中定义的
    print("ep_length",ep_length)


    # # 记忆模块相关参数
    # mem_kwargs ={
    #   'memory_size':FLAGS.memory_size
    # }
    # print("___________",**mem_kwargs.memory_size)
    #定义智能体
    agent =  GBMRagent.Agent(num_actions=env._num_actions,obs_size=75,memory_size=FLAGS.memory_size)

    agent.vae_initial()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # agent._vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())


    while True:
        observation, reward = env.reset()
        obs = observation.reshape(1,75).astype('float32') / 255 #这个应该放到函数里面
        state = agent.obs2state(obs)
        observations =[observation]
        epshistory = agent.EpsHistory_Initial()

        for tt in range(ep_length):
            #env.trender()#没有一个直观的显示
            #epshistory = agent.EpsHistory_add(state)
            #print(state.shape)#（1，32）
            action,readinfo= agent.infer(state,epshistory)
            #print(action,readinfo)
            #action= agent.TakeRandomAction()
            observation_, reward = env.step(action)
            obs_ = observation_.reshape(1,75).astype('float32') / 255 #这个应该放到函数里面
            state_ = agent.obs2state(obs_)
            epshistory = agent.EpsHistory_add([state,action,reward,state_])
            observation= observation_
            state= state_
            observations.append(observation)# 因为后面要以batch的形式输入，要把这个三维度数组变成四维的
            
            

        # 编码器训练，每个回合训练一次，也可以选择叠加的形式
        observations = np.stack(observations)
        agent.Memory_update(epshistory)
        agent.Memory_abstract()
        agent.Memory_reconstruct()
        input_train = observations.reshape(ep_length+1, observations.shape[1]*observations.shape[2]*observations.shape[3]).astype('float32') / 255     
        agent.vae_train(input_train, epochs=2, batch_size=64)
        # agent._vae.fit(input_train, input_train, epochs=1, batch_size=64)

if __name__ == '__main__':
  with tf.device('/cpu:0'):
    app.run(main)
    