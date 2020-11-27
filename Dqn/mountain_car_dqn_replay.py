from matplotlib import pyplot as plt
import random
from dqn import DQN
import torch
import gym
from collections import deque

env = gym.make("MountainCar-v0")


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function

def modify_reward(reward):
    mod_reward = reward + 0.5

    if reward >= 0.5:
        mod_reward += 100
    elif reward >= 0.25:
        mod_reward += 20
    elif reward >= 0.1:
        mod_reward += 10
    elif reward >= 0:
        mod_reward += 5

    return mod_reward

def q_learning(env, estimator: DQN, n_episode,
                gamma = 1.0, epsilon = 0.1, epsilon_decay = 0.99,
                replay_size = 20):

    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        is_done = False
        episode_memory = []
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)

            modified_reward = modify_reward(next_state[0])
            total_reward_episode[episode] += reward
            total_mod_reward_episode[episode] += modified_reward

            memory.append((state, action, next_state, modified_reward, is_done))
            episode_memory.append((state, action, next_state, modified_reward, is_done))

            if is_done:
                if total_reward_episode[episode] > -200: # nice episode to replay :)
                    total_memory.append(episode_memory)
                break

            estimator.replay(memory, replay_size, gamma)

            state = next_state
            
        print('Episode {}: reward: {}, epsilon: {}'.format(
            episode, total_reward_episode[episode], epsilon
        ))

        epsilon = max(epsilon * epsilon_decay, 0.01)

        if (episode % 10) == 1:
            total_wins = 0
            for test in range(100):
                total_wins += 1 if run_episode(env, estimator) > -200 else 0
            if (total_wins > 80):
                break

        if (episode % 100) == 1:
            for good_ep_mem in total_memory:
                estimator.replay(good_ep_mem, len(good_ep_mem), gamma)
            #by_value = sorted(total_memory, key=lambda x: x[3])
            #best_replay_size = (int(episode / 100) * 200) + replay_size
            #best_episodes = by_value[:best_replay_size]
            #estimator.replay(good_ep_mem, len(good_ep_mem), gamma)
            render_episode(env, estimator)
            
def render_episode(env, estimator: DQN):
    state = env.reset()
    is_done = False
    while not is_done:
        action = torch.argmax(estimator.predict(state)).item()
        state, reward, is_done, info = env.step(action)
        env.render()
    env.close()

def run_episode(env, estimator: DQN):
    state = env.reset()
    is_done = False
    total_reward = 0

    while not is_done:
        action = torch.argmax(estimator.predict(state)).item()
        state, reward, is_done, info = env.step(action)
        total_reward += reward

    return total_reward

n_episode = 10000
total_reward_episode = [0] * n_episode
total_mod_reward_episode = [0] * n_episode

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 50
lr = 0.001

memory = deque(maxlen=10000)
total_memory = deque(maxlen=1000)

dqn = DQN(n_state, n_action, n_hidden, lr)

q_learning(env, dqn, n_episode, gamma=0.9, epsilon=0.3, replay_size=20)

plt.plot(total_reward_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.plot(total_mod_reward_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

dqn.save()

print('rendering demo episodes... enter any char to stop')
while input() == '':
    render_episode(env, dqn)

