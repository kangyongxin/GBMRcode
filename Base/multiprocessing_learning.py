# 为了搞清楚a3c的实现方法，简单学习多线程,多核运算
import multiprocessing as mp
import threading as td
import time
import gym
import numpy as np

def job(q):
    res = 0
    for i in range(1000000):
        res += i + i**2 + i**3
    q.put(res) # queue

def multicore():
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q,))
    p2 = mp.Process(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print('multicore:',res1 + res2)

def normal():
    res = 0
    for _ in range(2):
        for i in range(1000000):
            res += i + i**2 + i**3
    print('normal:', res)


def multithread():
    q = mp.Queue() # thread可放入process同样的queue中
    t1 = td.Thread(target=job, args=(q,))
    t2 = td.Thread(target=job, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()
    print('multithread:', res1 + res2)

def job1(x):
    return x*x


def multicore1():
    pool = mp.Pool(processes=10)
    res = pool.map(job1, range(10))
    print(res)

class Agent():
    def __init__(self,env=None):
        self.num_actions = env.action_space.n
        

def env1(env_name):
    env = gym.make(env_name)
    obs = env.reset()
    num_actions = env.action_space.n
    sum_r=0
    while True:
        action = np.random.choice(list(range(num_actions)))
        obs_,rew,done,info = env.step(action)
        print(env_name,"reward",rew)
        sum_r += rew
        if done:
            print(env_name,"reward",sum_r)
            break
    return sum_r



def multienv():
    num_env = mp.cpu_count()-20
    input_name = ['MsPacman-v0' for i in range(num_env)]
    pool = mp.Pool(processes=num_env)
    res = pool.map(env1,input_name)
    print(res)
#MsPacman-v0

if __name__ == '__main__':
    st = time.time()
    normal()
    st1 = time.time()
    print('normal time:', st1 - st)
    multithread()
    st2 = time.time()
    print('multithread time:', st2 - st1)
    multicore1()
    st3 = time.time()
    print('multicore time:', st3 - st2)
    multienv()
