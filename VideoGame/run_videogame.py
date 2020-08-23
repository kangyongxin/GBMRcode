# 0821 kangyongxin
# VideoGame.md
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import curses

from absl import app
from absl import flags
import gym
from functools import reduce
import matplotlib.pyplot as plt

#from maze.maze_env20 import Maze
import GBMRAgent

FLAGS = flags.FLAGS
#环境相关
flags.DEFINE_string('env_name','maze','a name of env')
#调试函数接口
flags.DEFINE_boolean('print_functionname', True,
                     'Whether to print_functionname.')
flags.DEFINE_string('save_path','./RESULT/test/','save path')
# 智能体相关
flags.DEFINE_integer('memory_size', 1000,'the number of nodes we are able to store in the graph.')
flags.DEFINE_integer('memory_word_size', 2,'the lenth of words we are able to store in the graph.')

# 交互过程相关
flags.DEFINE_integer('episode_length', 1000,'the mux number of steps in a episode.')
flags.DEFINE_integer('num_episode', 40,'num episodes.')

def main(_):
    print("first come here",FLAGS.env_name)
    SAVEPATH = FLAGS.save_path
    ep_length = FLAGS.episode_length
    n_trj = FLAGS.num_episode
    if FLAGS.env_name=='maze':
        env = Maze()
        num_actions = env.n_actions
        observation = env.reset()
        print(observation)
        dim_obs = len(observation)
    else:
        env = gym.make(FLAGS.env_name)
        num_actions = env.action_space.n
        observation = env.reset()
        #print(observation)
        shape_obs = observation.shape
        dim_obs = reduce(lambda x,y: x*y, list(shape_obs)) 
    if FLAGS.print_functionname == True:
        print("ep_length",ep_length,"num_actions",num_actions,"dim_obs",dim_obs)
    # 智能体初始化
    agent = GBMRAgent.Agent(num_actions=num_actions,dim_obs=dim_obs,memory_size=FLAGS.memory_size,memory_word_size=FLAGS.memory_word_size)
    

    for eps in range(n_trj):
        observation = env.reset()
        step =0
        while step<ep_length:
            step += 1
            #state = agent.obs2state(observation)
            state = agent.obs_ram(observation)
            print(state)
            action = 4#agent.TakeRandomAction()
            observation_, reward,done,info = env.step(action)
            state_ = agent.obs_ram(observation_)
            env.render()
            observation= observation_
            agent.ExternalMemory.pairwriter([state,action,reward,state_])
            if done:
                print("done !",reward)      
                plt.figure(eps)
                agent.ExternalMemory.plotMemory()
                plt.savefig(SAVEPATH+str(eps)+'.png')
                plt.close(eps)
                break 
    

if __name__ == '__main__':
    app.run(main)

