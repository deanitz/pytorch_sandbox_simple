from os import stat
import gym
import torch
from collections import defaultdict

env = gym.make('CliffWalking-v0')

def generate_e_greedy_policy(n_action, epsilon):
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def q_learning(env, gamma, n_episode, alpha):

    Q = defaultdict(lambda: torch.zeros(n_action))
    

    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        while not is_done:
            action = e_greedy(state, Q)
            next_state, reward, is_done, info = env.step(action)
            td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * td_delta
            if is_done:
                break
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()

    return Q, policy

gamma = 1
n_episode = 500
alpha = 0.4
epsilon = 0.1
n_action = env.action_space.n
e_greedy = generate_e_greedy_policy(n_action, epsilon)

optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha)
print('optimal_policy')

for row in range(4):
    print('\n')
    for col in range(12):
        index = row * 12 + col
        action = optimal_policy.get(index)
        if action is None:
            action = 'x'
        print('{:2}: {}'.format(index, action), end=' |')