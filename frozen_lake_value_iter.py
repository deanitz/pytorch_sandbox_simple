import gym
import torch
import matplotlib.pyplot as plt

def plot_history(gammas, avg_reward):
    plt.plot(gammas, avg_reward)
    plt.title('total rewards depend on gamma')
    plt.xlabel('gamma')
    plt.ylabel('avg total reward')
    plt.show()

def value_iteration(env, gamma, threshold):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.empty(n_state)
        # state - индекс!
        for state in range(n_state):
            v_actions = torch.zeros(n_action)
            for action in range(n_action):
                for trans_prob, new_state, reward, _ in\
                    env.env.P[state][action]:

                    v_actions[action] +=\
                        trans_prob *\
                        (reward + gamma * V[new_state])
                
            V_temp[state] = torch.max(v_actions)

        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()

        if max_delta <= threshold:
            break
    return V

def policy_improvement(env, V, gamma):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    optimal_policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in\
                    env.env.P[state][action]:
                v_actions[action] +=\
                    trans_prob *\
                    (reward + gamma * V[new_state])
        optimal_policy[state] = torch.argmax(v_actions)

    return optimal_policy

def policy_eval(env, policy, gamma, threshold):
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = policy[state].item()
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_temp[state] += trans_prob *\
                    (reward + gamma * V[new_state])

        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()

        if max_delta <= threshold:
            break
    return V

def policy_iteration(env, gamma, threshold):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state,)).float()
    while True:
        V = policy_eval(env, policy, gamma, threshold)
        optim_policy = policy_improvement(env, V, gamma)
        if torch.equal(optim_policy, policy):
            return V, optim_policy
        policy = optim_policy


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
        state = new_state # we need to choose correct action on next iteration!!!!!!!
        total_reward += reward
        if is_done:
            break
    return total_reward

env = gym.make('FrozenLake-v0', map_name="8x8", is_slippery = True)
env.render()

# gammas = [0, 0.1, 0.2, 0.4, 0.8, 0.99, 1]
gammas = [ 0.1, 0.99 ]
avg_reward_gamma = []
threshold = 0.0001
n_episode = 10000

for gamma in gammas:
    # V_optimal = value_iteration(env, gamma, threshold)
    # print('V_optimal: ', V_optimal)
    # P_optimal = policy_improvement(env, V_optimal, gamma)
    # print('P_optimal: ', P_optimal)
    
    V_optimal, P_optimal = policy_iteration(env, gamma, threshold)
    print('V_optimal: ', V_optimal)
    print('P_optimal: ', P_optimal)
    total_rewards = []
    for episode in range(n_episode):
        total_reward = run_episode(env, P_optimal)
        total_rewards.append(total_reward)
    avg_reward_gamma.append(sum(total_rewards) / n_episode)

plot_history(gammas, avg_reward_gamma)
