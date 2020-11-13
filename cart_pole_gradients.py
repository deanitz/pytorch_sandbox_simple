import gym
import torch

env = gym.make('CartPole-v0')
clear = lambda: print("\033c")

n_state = env.observation_space.shape[0]
print(n_state)

n_action = env.action_space.n
print(n_action)

def run_episode(env, weight, render=False):
    step = 0
    state = env.reset()
    grads = []
    total_reward = 0
    is_done = False
    while (not is_done) or (render and step < 1000):
        step += 1
        if render:
            env.render()

        state = torch.from_numpy(state).float()
        pred = torch.matmul(state, weight)

        # Nonlinearity!
        probs = torch.nn.Softmax(dim=0)(pred)

        # if probs = [0.6, 0,4] => action = 1 in 60% cases
        action = int(torch.bernoulli(probs[1]).item())

        d_softmax = torch.diag(probs) - probs.view(-1, 1) * probs
        d_log = d_softmax[action] / probs[action]

        # Some magic with derivatives I dunno
        grad = state.view(-1, 1) * d_log
        grads.append(grad)

        state, reward, is_done, _ = env.step(action)
        total_reward += reward
    return total_reward, grads

n_episode = 1000
best_total_reward = 0
weight = torch.rand(n_state, n_action)
total_rewards = []
learn_rate = 0.001
n_episodes_actual = 0

for episode in range(n_episode):
    total_reward, gradients = run_episode(env, weight)
    print('Episode {}: {}'.format(episode, total_reward))
    n_episodes_actual += 1
    for i, gradient in enumerate(gradients):
        weight += learn_rate * gradient * (total_reward - i)

    total_rewards.append(total_reward)

    # if last 100 episodes reward was >= 195 - task is solved!
    if episode >= 99 and sum(total_rewards[-100:]) >= (195 * 100):
        break

n_episode_test = 200
total_rewards_test = []

import matplotlib.pyplot as plt
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

for episode in range(n_episode_test):
    total_reward, gradients = run_episode(env, weight)
    print('Episode {}: {}'.format(episode, total_reward))
    total_rewards_test.append(total_reward)

print('Avg total reward: {}, trained on {} episodes'.format(sum(total_rewards_test) / n_episode_test, n_episodes_actual))
run_episode(env, weight, render=True)