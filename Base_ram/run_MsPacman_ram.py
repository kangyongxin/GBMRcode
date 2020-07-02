import numpy as np 
from six.moves import range
from six.moves import zip
from absl import app
from absl import flags
#from envs.tpycolab import tenv as pycolab_env
import GBMRagent
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import gym
import gym
import time
from random import randint

'''
超参数读入：环境相关，流程相关，智能体相关
'''
FLAGS = flags.FLAGS
#调试函数接口
flags.DEFINE_boolean('print_functionname', True,
                     'Whether to print_functionname.')

# 智能体相关
flags.DEFINE_integer('memory_size', 1000,'the number of nodes we are able to store in the graph.')
flags.DEFINE_integer('memory_word_size', 2,'the lenth of words we are able to store in the graph.')

def main(_):
    if FLAGS.print_functionname == True:
        print("Hello world!")   
    env = gym.make("MsPacman-ram-v4")
    ep_length = 1000
    num_actions = env.action_space.n
    observation = env.reset()
    dim_obs = observation.shape
    if FLAGS.print_functionname == True:
        print("ep_length",ep_length,"num_actions",num_actions,"dim_obs",dim_obs)
    # 智能体初始化
    agent = GBMRagent.Agent(num_actions=num_actions,dim_obs=dim_obs,memory_size=FLAGS.memory_size,memory_word_size=FLAGS.memory_word_size)
    # agent.vae_initial()
    # agent.vaev_initial()

    ith_episode = 0 
    reward2step =[]
    while True:
        # 开始新的episode
        ith_episode += 1
        if FLAGS.print_functionname == True:
            print("ith_episode", ith_episode)
        
        observation = env.reset()
        #state = agent.obs2state(observation)
        state = agent.obs_ram(observation)
        observations =[]
        rewards =[]# 这个后来要用来算v值做监督信号
        epshistory = agent.EpsHistory_Initial()
        rew = 0

        # 环境交互
        for tt in range(ep_length):
            
            # if FLAGS.print_functionname == True:
            #     print("jth_step",tt)
            #action = agent.TakeRandomAction()
            #print("state",state)
            action, readinfo = agent.infer(state,epshistory)
            observation_, reward,done,info = env.step(action)
            #env.render()
            #state_ = agent.obs2state(observation_)
            if tt<90:
                continue
            state_ = agent.obs_ram(observation_)
            epshistory = agent.EpsHistory_add([state,action,reward,state_])
            observation= observation_
            state= state_
            observations.append(observation)
            rewards.append(reward)
            reward2step.append(reward)
            rew += reward
            if done or info['ale.lives']<3:
                print(ith_episode,"th episode with reward ",rew,"in ",tt," step")
                break
        # 记忆重构
        #print("history",epshistory)
        agent.Memory_update(epshistory)
        agent.Memory_abstract()
        agent.Memory_reconstruct()

        # 训练参数
        observations = np.stack(observations)
        rewards = np.stack(rewards)
        # input_train = observations.reshape(ep_length, observations.shape[1]*observations.shape[2]*observations.shape[3]).astype('float32') / 255     
        # rewards = rewards.reshape(ep_length,1).astype('float32') / 255 
        # agent.vaev_train(input_train,rewards,epochs =2, batch_size =64)
        agent.train_agg()
        
        print("write current data",len(reward2step))
        np.save("rewstep.npy",reward2step)
    


if __name__ == '__main__':
  with tf.device('/cpu:0'):
    app.run(main)

