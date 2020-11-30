import sys
from time import sleep
import gym
from gym.wrappers.time_limit import TimeLimit
import torch
from collections import deque
from dqn_cnn import CNN_DQN
import numpy as np


env = gym.make('PongDeterministic-v4')

ACTIONS = [0, 2, 3]
n_action = len(ACTIONS)

state_shape = env.observation_space.shape

print(env.unwrapped.get_action_meanings())
print(n_action)
print(state_shape)

import torchvision.transforms as imgt
from PIL import Image
image_size = 84
transform = imgt.Compose([
    imgt.ToPILImage(),
    imgt.Resize((image_size, image_size), interpolation=Image.CUBIC),
    imgt.ToTensor()
])

def get_state(obs):
    state = obs.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = transform(state).unsqueeze(0)
    return state

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

def render_episode(env: TimeLimit, estimator: CNN_DQN):
    obs = env.reset()
    state = get_state(obs)
    is_done = False
    while not is_done:
        sleep(0.0415) # (~24 fps)

        rgb = env.render('rgb_array')
        upscaled=repeat_upsample(rgb, 3, 4)
        viewer.imshow(upscaled)

        actionIndex = torch.argmax(estimator.predict(state)).item()
        action = ACTIONS[actionIndex]
        obs, reward, is_done, _ = env.step(action)
        if reward != 0:
            print(reward)
        state = get_state(obs)
    env.close()

ai = CNN_DQN(3, n_action, False)
ai.load()

while True:
    print('rendering loaded model episode... enter to continue, any char to stop')
    render_episode(env, ai)
    if input() != '':
        break