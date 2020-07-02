# testing ....


'''
智能体与环境交互的基本流程

整体分为四部分： 参数读入与初始化， 环境交互， 记忆重构， 参数训练

用a2c实现对比策略
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
import gym
import logging

'''
超参数读入：环境相关，流程相关，智能体相关
'''
FLAGS = flags.FLAGS
# 环境相关

# flags.DEFINE_enum('pycolab_game', 'key_to_door',
#                   ['key_to_door', 'active_visual_match'],
#                   'The name of the game in pycolab environment')
# flags.DEFINE_integer('pycolab_num_apples', 10,
#                      'Number of apples to sample from the distractor grid.')
# flags.DEFINE_float('pycolab_apple_reward_min', 1.,
#                    'A reward range [min, max) to uniformly sample from.')
# flags.DEFINE_float('pycolab_apple_reward_max', 10.,
#                    'A reward range [min, max) to uniformly sample from.')
# flags.DEFINE_boolean('pycolab_fix_apple_reward_in_episode', True,
#                      'Fix the sampled apple reward within an episode.')
# flags.DEFINE_float('pycolab_final_reward', 10.,
#                    'Reward obtained at the last phase.')
# flags.DEFINE_boolean('pycolab_crop', True,
#                      'Whether to crop observations or not.')

# 尝试MsPacman
# flags.DEFINE_enum('atari_game',
#                   'MsPacman-v0',
#                   'The name of the game in atari')

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
    # env_kwargs = {
    #   'game': FLAGS.pycolab_game,
    #   'num_apples': FLAGS.pycolab_num_apples,
    #   'apple_reward': [FLAGS.pycolab_apple_reward_min,
    #                    FLAGS.pycolab_apple_reward_max],
    #   'fix_apple_reward_in_episode': FLAGS.pycolab_fix_apple_reward_in_episode,
    #   'final_reward': FLAGS.pycolab_final_reward,
    #   'crop': FLAGS.pycolab_crop
    #   }
    # env_kwargs = {
    #   'game': FLAGS.pycolab_game
    #   }
    # if FLAGS.print_functionname == True:
    #     print("env_kwargs: ",env_kwargs)
    # env_builder = pycolab_env.PycolabEnvironment
    # env=env_builder(**env_kwargs)  #以字典的形式传递参数，方便函数内部对参数的分别引用
    #env = gym.make("MsPacman-v0")
    # env = gym.make("CartPole-v1")
    env = gym.make("Alien-v0")
    # ep_length = env.episode_length# 在key_to_door的环境中定义的
    ep_length = 1000
    #num_actions = env.num_actions
    num_actions = env.action_space.n
    observation = env.reset()
    dim_obs = observation.shape
    if FLAGS.print_functionname == True:
        print("ep_length",ep_length,"num_actions",num_actions,"dim_obs",dim_obs)

    # 智能体初始化
    agent = GBMRagent.Agent(num_actions=num_actions,dim_obs=dim_obs,memory_size=FLAGS.memory_size,memory_word_size=FLAGS.memory_word_size)
    # agent.vae_initial()
    agent.vaev_initial()

    ith_episode = 0 
    reward2step =[]
    ep_rews = [0.0]
    while True:
        # 开始新的episode
        ith_episode += 1
        if FLAGS.print_functionname == True:
            print("ith_episode", ith_episode)
        
        observation = env.reset()
        state = agent.obs2state(observation)
        observations =[]
        rewards =[]# 这个后来要用来算v值做监督信号
        epshistory = agent.EpsHistory_Initial()
        ep_rews.append(0.0)
        values =[]
        dones=[]
        actions=[]
        states =[]
        # 环境交互
        for tt in range(ep_length):
            # if FLAGS.print_functionname == True:
            #     print("jth_step",tt)
            #action = agent.TakeRandomAction()
            #action, readinfo = agent.infer(state,epshistory)
            #action = agent.inferModelfree(state)
            action,value=agent.a2cmodel.action_value(state)
            observation_, reward,done,_ = env.step(action)
            #env.render()
            state_ = agent.obs2state(observation_)
            epshistory = agent.EpsHistory_add([state,action,reward,state_])
            observation= observation_
            state= state_
            observations.append(observation)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            actions.append(action)
            states.append(state)
            reward2step.append(reward)
            ep_rews[-1] += reward
            if done:
                break
        logging.info("Episode: %03d, Reward: %05d" % (len(ep_rews)-1, ep_rews[-2]))
        # 记忆重构
        # agent.Memory_update(epshistory)
        # agent.Memory_abstract()
        # agent.Memory_reconstruct()
        # a2c 的训练
        #print("ep_rews",ep_rews)#只有一个数
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        values = np.stack(values)
        actions = np.stack(actions)
        #print("states",states)
        states = np.stack(states)
        values= np.squeeze(values)
        states = np.squeeze(states)
        #print("rewards",rewards,"dons",dones,"values",values,"actions",actions,"states",states)
        
        _, next_value = agent.a2cmodel.action_value(state_)
        returns, advs = agent._returns_advantages(rewards, dones, values, next_value)
        acts_and_advs = np.concatenate([actions[:,None], advs[:,None]], axis=-1)
        #print("next_value",next_value,"returns",returns,"advs",advs,"advs,acts_and_advs",acts_and_advs)
        a2closses = agent.a2cmodel.train_on_batch(states, [acts_and_advs, returns])
        logging.info("[%d/%d] Losses: %s" % (ith_episode+1, ith_episode, a2closses))
        # _, next_value = agent.a2cmodel.action_value(state_[None, :])
        # returns, advs = agent._returns_advantages(rewards, dones, values, next_value)
        # acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        # a2closses = agent.a2cmodel.train_on_batch(states, [acts_and_advs, returns])
        # logging.debug("[%d/%d] Losses: %s" % (ith_episode+1, ith_episode, a2closses))
        
        # 训练参数
        observations = np.stack(observations)
        rewards = np.stack(rewards)
        print("obs shape",observations.shape)
        # input_train = observations.reshape(ep_length, observations.shape[1]*observations.shape[2]*observations.shape[3]).astype('float32') / 255     
        # rewards = rewards.reshape(ep_length,1).astype('float32') / 255 
        input_train = observations.reshape(tt+1, observations.shape[1]*observations.shape[2]*observations.shape[3]).astype('float32') / 255     
        rewards = rewards.reshape(tt+1,1).astype('float32') / 255 
        agent.vaev_train(input_train,rewards,epochs =2, batch_size =64)
        # agent.train_agg()
        #if len(reward2step)%10==0:
        print("write current data",len(reward2step))
        np.save("rewstep.npy",reward2step)
        

if __name__ == '__main__':
  with tf.device('/gpu:0'):
    app.run(main)
