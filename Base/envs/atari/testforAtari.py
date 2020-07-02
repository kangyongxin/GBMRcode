

# 测试 Atari 环境的入口代码


import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from common.vec_env import VecFrameStack, VecNormalize, VecEnv
from common.vec_env.vec_video_recorder import VecVideoRecorder
from common.tf_util import get_session
from common.atari_wrappers import make_atari, wrap_deepmind

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def testenvs():
    #env = gym.make("CartPole-v1")
    env = gym.make("MsPacman-v0")
    #env = gym.make("MsPacMan-v0")
    observation = env.reset()
    print("observation",observation.shape)
    for _ in range(100):
        env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print("reward",reward)
        if done:
            observation = env.reset()
    env.close()

def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            print("are we here?")
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env

def testbaselines(args):
    # 用baseline对环境的包装方法，尝试运行最简代码
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    #extra_args = parse_cmdline_kwargs(unknown_args)

    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))
    env = make_atari(env_id)
    #env = build_env(args)
    print("env builded ",env)
    obs = env.reset()
    reset = True

    print("env reseted")
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        new_obs, rew, done, _ = env.step(action)
        obs = new_obs
        if done:
            print(done)
            obs = env.reset()
            reset = True
def testt2t(args):
    pass


def main(args):
    testenvs()
    #testbaselines(args)


if __name__ == '__main__':
    print("first come here")
    from gym import envs
    print(envs.registry.all())
    main(sys.argv)

