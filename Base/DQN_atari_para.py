# 一个基础比较的算法，主要目的是给Atari 游戏跑一个最基本的参考曲线
# 从ram 出发
# 列出所有可调参数， 通过命令文件读入，进行调参对比
# 输出结果也要单独放，方便对比
import gym
import time
from random import randint
from DQNAgent import DQNAgent
import numpy as np
from absl import app
from absl import flags
import tensorflow as tf
#参数列表 

FLAGS = flags.FLAGS
flags.DEFINE_string('path_name','./results/1','a name to save result')
flags.DEFINE_enum('specific_device','/cpu:0',['/cpu:0','/cpu:10','/gpu:0'],'specific device to run')
flags.DEFINE_enum('env_name','MsPacman-ram-v0',['MsPacman-ram-v0','Alien-ram-v0'],'the name of game in atari')
flags.DEFINE_float('learning_rate',0.01,'learning rate for BP ')
flags.DEFINE_float('reward_decay',0.9,'gamma factor in Q learning ')
flags.DEFINE_float('e_greedy',0.9,'epsilon for greedy search')
flags.DEFINE_integer('replace_target_iter',300,'replace target net para by eval net')
flags.DEFINE_integer('memory_size',2000,'size of replay buffer')
flags.DEFINE_integer('batch_size',256,'batch size for training')
flags.DEFINE_integer('max_step',1000,'max step for each episode')
flags.DEFINE_boolean('display',False,'show the process of training')
#env = gym.make("Alien-ram-v0")
#env = gym.make("MsPacman-ram-v0")
def main(_):
    env = gym.make(FLAGS.env_name)

    num_actions = env.action_space.n
    num_features = env.observation_space.shape[0]
    print("num_actions",num_actions,"num_features",num_features)

    agent = DQNAgent(
        num_actions,
        num_features,
        learning_rate= FLAGS.learning_rate,
        reward_decay=FLAGS.reward_decay,
        e_greedy=FLAGS.e_greedy,
        replace_target_iter=FLAGS.replace_target_iter,
        memory_size=FLAGS.memory_size,
        batch_size=FLAGS.batch_size,
        e_greedy_increment=None,
    )
    eps=0
    score_list=[]
    while True:
        eps += 1
        obs = env.reset()
        obs = obs.reshape(-1,num_features)
        score = 0
        for step in range(FLAGS.max_step):
            #print("step",step)
            #action = randint(0,num_actions-1)
            # action = agent.take_random_action()
            action = agent.choose_action(obs)
            obs_,rew,done,info = env.step(action)
            score += rew
            obs_ = obs_.reshape(-1,num_features)
            agent.store_transition(obs,action,rew,obs_)
            agent.train_model()
            if FLAGS.display:
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
                np.save(FLAGS.path_name+'dqn_mspacma.npy',score_list)
                break
        if np.mean(score_list[-10:]) > 1000:
            agent.save_model(FLAGS.path_name+'dqn_mspacman.h5')
            break

if __name__=='__main__': 
    with tf.device('/gpu:3'):
        app.run(main)
    #当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。