from time import sleep
import gym
import torch
import random

from collections import deque
import copy

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the renderer as it's what we'll be using to plot 
from gym.envs.classic_control import rendering
def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print('Number of repeats must be larger than 0, k: {}, l: {}, returning default array!'.format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

viewer = rendering.SimpleImageViewer()

env = gym.make('PongDeterministic-v4')

ACTIONS = [0, 2, 3]
# ACTIONS = [0, 1, 2, 3, 4, 5]
n_action = len(ACTIONS)

state_shape = env.observation_space.shape

print(env.unwrapped.get_action_meanings())
print(n_action)
print(state_shape)

env.reset()

is_done = False
obs = []
while not is_done:
    sleep(0.0415) # (~24 fps)

    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb, 3, 4)
    viewer.imshow(upscaled)

    action = ACTIONS[random.randint(0, n_action - 1)]
    obs, reward, is_done, _ = env.step(action)
    if reward != 0:
        print(reward)

print(obs)

env.close()

