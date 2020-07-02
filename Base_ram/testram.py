# 测试如何从ram中读取数据


import gym
import time
from random import randint
#env = gym.make("Alien-ram-v0")
env = gym.make("MsPacman-ram-v0")

num_actions = env.action_space.n
obs_space = env.observation_space
print("num_actions",num_actions,"obs_space",obs_space)
eps=0
while True:
    eps += 1
    obs = env.reset()
    for step in range(1000):
        print("step",step)
        action = randint(0,num_actions-1)
        obs_,rew,done,info = env.step(action)
        env.render()
        print("x :",obs_[10],"y :",obs_[16])
        # if step<90:
        #     continue
        # if step>160:
        #     break

        delta_obs =obs-obs_
        obs=obs_
        #print('delta_obs',delta_obs,'rew',rew,'done',done,'info',info)
        #print('delta_obs\n',delta_obs)
        time.sleep(0.1)
        if done or info['ale.lives']<3:
            print(eps,"th episode with reward ",rew)
            break