'''
智能体与环境交互的基本流程

整体分为四部分： 参数读入与初始化， 环境交互， 记忆重构， 参数训练
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
import matplotlib.pyplot as plt

'''
超参数读入：环境相关，流程相关，智能体相关
'''
FLAGS = flags.FLAGS
# 环境相关

flags.DEFINE_enum('pycolab_game', 'key_to_door',
                  ['key_to_door', 'active_visual_match'],
                  'The name of the game in pycolab environment')
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

# 流程相关

flags.DEFINE_boolean('print_functionname', True,
                     'Whether to print_functionname.')

# 智能体相关

flags.DEFINE_integer('memory_size', 1000,'the number of nodes we are able to store in the graph.')
flags.DEFINE_integer('memory_word_size', 32,'the lenth of words we are able to store in the graph.')


def main(_):
    if FLAGS.print_functionname == True:
        print("Hello world!")

    # 环境初始化
    env_kwargs = {
      'game': FLAGS.pycolab_game,
      'num_apples': FLAGS.pycolab_num_apples,
      'apple_reward': [FLAGS.pycolab_apple_reward_min,
                       FLAGS.pycolab_apple_reward_max],
      'fix_apple_reward_in_episode': FLAGS.pycolab_fix_apple_reward_in_episode,
      'final_reward': FLAGS.pycolab_final_reward,
      'crop': FLAGS.pycolab_crop
      }
    if FLAGS.print_functionname == True:
        print("env_kwargs: ",env_kwargs)
    env_builder = pycolab_env.PycolabEnvironment
    env=env_builder(**env_kwargs)  #以字典的形式传递参数，方便函数内部对参数的分别引用
    ep_length = env.episode_length# 在key_to_door的环境中定义的
    num_actions = env.num_actions
    dim_obs = env.observation_shape
    if FLAGS.print_functionname == True:
        print("ep_length",ep_length,"num_actions",num_actions,"dim_obs",dim_obs)

    # 智能体初始化
    agent = GBMRagent.Agent(num_actions=num_actions,dim_obs=dim_obs,memory_size=FLAGS.memory_size,memory_word_size=FLAGS.memory_word_size)
    # agent.vae_initial()
    agent.vaev_initial()

    ith_episode = 0 
    reward2step =[]
    while True:
        # 开始新的episode
        ith_episode += 1
        if FLAGS.print_functionname == True:
            print("ith_episode", ith_episode)
        
        observation, reward = env.reset()
        state = agent.obs2state(observation)
        observations =[observation]
        rewards =[reward]# 这个后来要用来算v值做监督信号
        epshistory = agent.EpsHistory_Initial()
        

        # 环境交互
        for tt in range(ep_length):
            # if FLAGS.print_functionname == True:
            #     print("jth_step",tt)
            #action = agent.TakeRandomAction()
            action, readinfo = agent.infer(state,epshistory)
            observation_, reward = env.step(action)
            state_ = agent.obs2state(observation_)
            print("--------------++++++++++++++++++,state",state)
            epshistory = agent.EpsHistory_add([state,action,reward,state_])
            observation= observation_
            state= state_
            observations.append(observation)
            rewards.append(reward)
            reward2step.append(reward)

        # 记忆重构
        print("-----------",epshistory)
        agent.Memory_update(epshistory)
        agent.Memory_abstract()
        agent.Memory_reconstruct()

        # 训练参数
        observations = np.stack(observations)
        rewards = np.stack(rewards)
        input_train = observations.reshape(ep_length+1, observations.shape[1]*observations.shape[2]*observations.shape[3]).astype('float32') / 255     
        rewards = rewards.reshape(ep_length+1,1).astype('float32') / 255 
        agent.vaev_train(input_train,rewards,epochs =2, batch_size =64)
        agent.train_agg()
        if len(reward2step)%100==0:
            print("write current data",len(reward2step))
            np.save("rewstep.npy",reward2step)

if __name__ == '__main__':
  with tf.device('/gpu:0'):
    app.run(main)
