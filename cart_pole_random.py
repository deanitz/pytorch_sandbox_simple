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
    total_reward = 0
    is_done = False
    while (not is_done) or (render and step < 500):
        step += 1
        if render:
            env.render()

        state = torch.from_numpy(state).float()
        pred = torch.matmul(state, weight)
        action = torch.argmax(pred)
        state, reward, is_done, _ = env.step(action.item())
        total_reward += reward
    return total_reward

n_episode = 1000
best_total_reward = 0
best_weight = 0
total_rewards = []
noise_scale = 0.01
n_episodes_actual = 0

for episode in range(n_episode):
    weight = best_weight + (noise_scale * torch.rand(n_state, n_action))
    total_reward = run_episode(env, weight)
    print('Episode {}: {}'.format(episode, total_reward))
    n_episodes_actual += 1
    if total_reward >= best_total_reward:
        best_weight = weight
        best_total_reward = total_reward
        noise_scale = max(noise_scale / 2, 1e-4)
    else:
        noise_scale = min(noise_scale * 2, 2)
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
    total_reward = run_episode(env, best_weight)
    print('Episode {}: {}'.format(episode, total_reward))
    total_rewards_test.append(total_reward)

print('Avg total reward: {}, trained on {} episodes'.format(sum(total_rewards_test) / n_episode_test, n_episodes_actual))
run_episode(env, best_weight, render=True)