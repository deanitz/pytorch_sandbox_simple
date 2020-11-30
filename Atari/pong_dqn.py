import sys
from time import sleep
import gym
from gym.wrappers.time_limit import TimeLimit
import torch
from collections import deque
from dqn_cnn import CNN_DQN


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

def q_learning(env: TimeLimit, estimator: CNN_DQN, n_episode,
                target_update_every = 10,
                gamma = 1.0, epsilon = 0.1, epsilon_decay = 0.99,
                replay_size = 32):

    step = 0
    for episode in range(n_episode):

        policy = estimator.gen_epsilon_greedy_policy(epsilon, n_action)
        obs = env.reset()
        state = get_state(obs)
        is_done = False
        while not is_done:

            actionAi = policy(state)
            actionGym = ACTIONS[actionAi]

            next_obs, reward, is_done, _ = env.step(actionGym)
            next_state = get_state(next_obs)

            total_reward_episode[episode] += reward

            memory.append((state, actionAi, next_state, reward, is_done))

            if is_done:
                break

            estimator.replay(memory, replay_size, gamma)

            state = next_state

            step += 1
            sys.stdout.write("                                                                                                                  \r"\
                + 'step {}'.format(step))
            
        print('Episode {}: reward: {}, epsilon: {}'.format(
            episode, total_reward_episode[episode], epsilon
        ))

        epsilon = max(epsilon * epsilon_decay, 0.01)

        if (episode % target_update_every) == 1:
            
            # update targets NN
            estimator.copy_target()
            estimator.save()

        if (episode % 100) == 1:
            # render_episode(env, estimator)

            # check if NN is well trained
            total_wins = 0
            for test in range(100):
                total_wins += 1 if run_episode(env, estimator) > 0 else 0
            if (total_wins > 90):
                estimator.copy_target()
                estimator.save()
                print('Finished training due to successful model')
                break

def render_episode(env: TimeLimit, estimator: CNN_DQN):
    obs = env.reset()
    state = get_state(obs)
    is_done = False
    while not is_done:
        sleep(0.0415) # (~24 fps)

        actionIndex = torch.argmax(estimator.predict(state)).item()
        action = ACTIONS[actionIndex]
        obs, reward, is_done, info = env.step(action)
        state = get_state(obs)
        env.render()
    env.close()

def run_episode(env: TimeLimit, estimator: CNN_DQN):
    obs = env.reset()
    state = get_state(obs)
    is_done = False
    total_reward = 0

    while not is_done:
        actionIndex = torch.argmax(estimator.predict(state)).item()
        action = ACTIONS[actionIndex]
        obs, reward, is_done, info = env.step(action)
        state = get_state(obs)
        total_reward += reward

    return total_reward

n_episode = 2000
total_reward_episode = [0] * n_episode

lr = 0.00025

ai = CNN_DQN(3, n_action, True, lr)
ai.load()

memory = deque(maxlen=100000)

q_learning(env, ai, n_episode, epsilon=0.5)

