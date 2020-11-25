from dqn import DQN
import torch
import gym
import numpy as np

np.random.seed(1)

env = gym.make("MountainCar-v0")

def render_episode(env, estimator: DQN):
    state = env.reset()
    is_done = False
    while not is_done:
        action = torch.argmax(estimator.predict(state)).item()
        state, reward, is_done, info = env.step(action)
        env.render()
    env.close()

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 50
lr = 0.001

dqn = DQN(n_state, n_action, n_hidden, lr)

dqn.load()

while True:
    print('rendering loaded model episode... enter to continue, any char to stop')
    render_episode(env, dqn)
    if input() != '':
        break