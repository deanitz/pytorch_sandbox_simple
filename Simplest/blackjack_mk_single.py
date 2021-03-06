import torch
import gym
from collections import defaultdict

env = gym.make('Blackjack-v0')

def run_episode(env, Q, n_action):
    """
    Выполняет эпизод, руководствуясь заданной Q-функцией
    @param env: среда OpenAI gym
    @param Q: Q-функция
    @param n_action: размерность пространства действий
    @return: списки состояний, действий и вознаграждений для всего эпизода
    """

    rewards = []
    actions = []
    states = []

    state = env.reset()
    action = torch.randint(0, n_action, [1]).item()

    is_done = False

    while not is_done:
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        if is_done:
            break
        action = torch.argmax(Q[state]).item()

    return states, actions, rewards

def mc_control_on_polcy(env, gamma, n_episode):

    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))

    for episode in range(n_episode):
        states_t, actions_t, rewards_t = run_episode(env, Q, n_action)
        return_t = 0
        G = {}

        for state_t, action_t, reward_t in zip(states_t[::-1],
                                                actions_t[::-1],
                                                rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        
        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]

    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()

    return Q, policy

def simulate_episode(env, policy):
    state = env.reset()
    is_done = False
    while not is_done:
        action = policy[state]
        state, reward, is_done, info = env.step(action)
        if is_done:
            return reward

gamma = 1
n_episodes = 1_000_000

optimal_Q, optimal_policy = mc_control_on_polcy(env, gamma, n_episodes)
for s in sorted(optimal_policy):
    print(s, optimal_policy[s])

print(len(optimal_policy))

hold_score = 18
hold_policy = {}
for player in range(2, 22):
    for dealer in range(1, 11):
        action = 1 if player <= hold_score else 0
        hold_policy[(player, dealer, False)] = action
        hold_policy[(player, dealer, True)] = action

n_episodes = 100_000
n_win_optimal = 0
n_win_simple = 0
n_lose_optimal = 0
n_lose_simple = 0

for _ in range(n_episodes):
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_optimal += 1
    elif reward == -1:
        n_lose_optimal += 1
    
    reward = simulate_episode(env, hold_policy)
    if reward == 1:
        n_win_simple += 1
    elif reward == -1:
        n_lose_simple += 1

print('Вероятность выигрыша при простой стратегии: {}'.format(n_win_simple / n_episodes))
print('Вероятность выигрыша при оптимальной стратегии: {}'.format(n_win_optimal / n_episodes))
print('Вероятность проигрыша при простой стратегии: {}'.format(n_lose_simple / n_episodes))
print('Вероятность проигрыша при оптимальной стратегии: {}'.format(n_lose_optimal / n_episodes))



