# 测试结果
import time
import gym
import numpy as np
from tensorflow.keras import models
env = gym.make('MsPacman-ram-v0')
model = models.load_model('dqn_mspacman.h5')
s = env.reset()
s = s.reshape(-1,env.observation_space.shape[0])
score = 0
while True:
    #env.render()
    time.sleep(0.01)
    a = np.argmax(model.predict(s))
    s, reward, done, _ = env.step(a)
    s = s.reshape(-1,env.observation_space.shape[0])
    score += reward
    if done:
        print('score:', score)
        break
env.close()