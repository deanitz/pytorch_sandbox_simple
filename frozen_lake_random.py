import gym
import torch

env = gym.make('FrozenLake-v0')

n_state = env.observation_space.n
print(n_state)

n_action = env.action_space.n
print(n_action)


env.render()

'''
@param env: gym environment
@param policy: таблица где состоянию соответствует действие
@return: total reward
'''
def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item() #.item() takes scalar value from tensor
        new_state, reward, is_done, info = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward
        
n_episodes = 1000
total_rewards = []

best_policy = torch.tensor(0)
best_reward = 0

for episode in range(n_episodes):
    random_policy = torch.randint(high=n_action, size=(n_state, ))
    total_reward = run_episode(env, random_policy)
    if total_reward > best_reward:
        best_reward = total_reward
        best_policy = random_policy.clone()
    total_rewards.append(total_reward)

print('Среднее случайное вознаграждение при случайной стратегии: {}'\
    .format(sum(total_rewards) / n_episodes))

n_episodes = 100
for episode in range(n_episodes):
    total_reward = run_episode(env, best_policy)
    print('Эпизод {}, вознаграждение: {}'.format(episode, total_reward))