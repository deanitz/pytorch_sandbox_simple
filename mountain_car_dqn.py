from matplotlib import pyplot as plt
import random
from dqn import DQN
import torch
import gym

env = gym.envs.make("MountainCar-v0")


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
        mod_reward += 10
    elif reward >= 0.1:
        mod_reward += 5
    elif reward >= 0:
        mod_reward += 1

    return mod_reward

def q_learning(env, estimator, n_episode,
                gamma = 1.0, epsilon = 0.1, epsilon_decay = 0.99
                ):

    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward

            modified_reward = modify_reward(next_state[0])
            total_mod_reward_episode[episode] += modified_reward

            q_values = estimator.predict(state).tolist()

            if is_done:
                q_values[action] = modified_reward
                estimator.update(state, q_values)
                break

            q_values_next = estimator.predict(next_state)
            
            q_values[action] = modified_reward + gamma * torch.max(q_values_next).item()

            estimator.update(state, q_values)

            state = next_state
            
        print('Episode {}: reward: {}, epsilon: {}'.format(
            episode, total_reward_episode[episode], epsilon
        ))

        epsilon = max(epsilon * epsilon_decay, 0.01)

n_episode = 1000
total_reward_episode = [0] * n_episode
total_mod_reward_episode = [0] * n_episode

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 50
lr = 0.001

dqn = DQN(n_state, n_action, n_hidden, lr)

q_learning(env, dqn, n_episode, gamma=0.99, epsilon=0.3)

plt.plot(total_reward_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.plot(total_mod_reward_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
