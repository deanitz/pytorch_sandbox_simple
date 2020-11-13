import torch
import gym
from time import sleep
import os

clear = lambda: print("\033c")

env = gym.make('SpaceInvaders-v0')
env.reset()

is_done = False
while not is_done:
    action = env.action_space.sample()
    sleep(0.0417) # (24 fps)
    new_state, reward, is_done, info = env.step(action)
    clear()
    print(info)
    print(new_state.shape)
    env.render()

