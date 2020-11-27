import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from windy_gridworld import WindyGridwoldEnv

env = WindyGridwoldEnv()

env.reset()
env.render()

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
            episode_length[episode] += 1
            episode_reward[episode] += reward
            if is_done or episode_reward[episode] < -200: # added exit on big fail
                break
            state = next_state

    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()

    return Q, policy

def sarsa_learning(env, gamma, n_episode, alpha):

    Q = defaultdict(lambda: torch.zeros(n_action))
    
    for episode in range(n_episode):
        state = env.reset()
        is_done = False

        action = e_greedy(state, Q)
        
        while not is_done:
            
            next_state, reward, is_done, info = env.step(action)
            next_action = e_greedy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            episode_length[episode] += 1
            episode_reward[episode] += reward
            if is_done:
                break
            state = next_state
            action = next_action

    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()

    return Q, policy





n_episode = 1000
episode_reward = [0] * n_episode
episode_length = [0] * n_episode

gamma = 1
alpha = 0.4
epsilon = 0.1
n_action = env.action_space.n
e_greedy = generate_e_greedy_policy(n_action, epsilon)

optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha)
print('optimal_policy')

shape = (7, 10)
for row in range(shape[0]):
    print()
    for col in range(shape[1]):
        index = row * shape[1] + col
        action = optimal_policy.get(index)
        if action is None:
            action = 'x'
        else:
            action = '↑→↓←'[action]
        #print('{:2}: {}'.format(index, action), end=' |')
        print(action, end='  ')

plt.plot(episode_length)
plt.title('Episode length')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.show()

plt.plot(episode_reward)
plt.title('Episode reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()