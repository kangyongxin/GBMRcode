# 测试如何从ram中读取数据


import gym
from random import randint
env = gym.make("Alien-ram-v0")

obs = env.reset()
num_actions = env.action_space.n
i=0
while True:
    i += 1
    action = randint(0,num_actions-1)
    obs,rew,done,info = env.step(action)
    env.render()
    print('obs',obs,'rew',rew,'done',done,'info',info)
    if done:
        obs = env.reset()
        print(i,"th episode with reward ",rew)