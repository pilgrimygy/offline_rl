import gym
import numpy as np


MUJOCO_SINGLE_ENVS = [
    "halfcheetah-medium-v2",
    ]
ATARI_ENVS = ['']


def get_env(env_name, **kwargs):
    if env_name in MUJOCO_SINGLE_ENVS:
        return gym.make(env_name, **kwargs)
    else:
        print("Env {} not found".format(env_name))
        exit(0)