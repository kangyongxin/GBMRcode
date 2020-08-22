# 一个基础比较的算法，主要目的是给Atari 游戏跑一个最基本的参考曲线
# 从ram 出发

import gym
import time
from random import randint
from DQNAgent import DQNAgent
import numpy as np

#env = gym.make("Alien-ram-v0")
env = gym.make("MsPacman-ram-v0")

num_actions = env.action_space.n
num_features = env.observation_space.shape[0]
print("num_actions",num_actions,"num_features",num_features)

agent = DQNAgent(num_actions,num_features)

eps=0
score_list=[]
while True:
    eps += 1
    obs = env.reset()
    obs = obs.reshape(-1,num_features)
    score = 0
    for step in range(1000):
        #print("step",step)
        #action = randint(0,num_actions-1)
        # action = agent.take_random_action()
        action = agent.choose_action(obs)
        obs_,rew,done,info = env.step(action)
        score += rew
        obs_ = obs_.reshape(-1,num_features)
        agent.store_transition(obs,action,rew,obs_)
        agent.train_model()
        env.render()
        #print("x :",obs_[10],"y :",obs_[16])
        if step<80:
             continue
        # if step>160:
        #     break

        #delta_obs =obs-obs_
        obs=obs_
        #print('delta_obs',delta_obs,'rew',rew,'done',done,'info',info)
        #print('delta_obs\n',delta_obs)
        #time.sleep(0.1)
        if done or info['ale.lives']<3:
            print(eps,"th episode with reward ",score)
            score_list.append(score)
            np.save('dqn_mspacma.npy',score_list)
            break
    if np.mean(score_list[-10:]) > 500:
        agent.save_model('dqn_mspacman.h5')
        break