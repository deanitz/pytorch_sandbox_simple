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

gamma = 1
n_episodes = 500_000

optimal_Q, optimal_policy = mc_control_on_polcy(env, gamma, n_episodes)
print(optimal_policy)
