import torch
import matplotlib.pyplot as plt

def plot_history(V_his, gamma):
    s0, = plt.plot([v[0] for v in V_his])
    s1, = plt.plot([v[1] for v in V_his])
    s2, = plt.plot([v[2] for v in V_his])

    plt.title('Optimal strategy with gamma = {}'.format(str(gamma)))
    plt.xlabel('Iteration')
    plt.ylabel('Values of states')
    plt.legend([s0, s1, s2],
                ["State s0", "State s1", "State s2"],
                loc="upper left")
    plt.show()

T = torch.tensor([[[0.8, 0.1, 0.1],
                   [0.1, 0.6, 0.3]],
                  [[0.7, 0.2, 0.1],
                   [0.1, 0.8, 0.1]],
                  [[0.6, 0.2, 0.2],
                   [0.1, 0.4, 0.5]]])

R = torch.tensor([1., 0., -1.])
gamma = 0.5
treshold = 0.0001

policy_optimal = torch.tensor([[1.0, 0.0],
                               [1.0, 0.0],
                               [1.0, 0.0]])

def policy_eval(policy, trans_matrix, rewards, gamma, treshold):
    """
    @param policy: матрица, содержащая вероятности выбора действий в каждом состоянии
    @param trans_matrix: матрица переходов
    @param rewards: вознаграждения в каждом состоянии
    @param gamma: дискаунтер - коэффициент обесценивания
    @param treshold: порог изменения ценности в итерации для выхода
    @return: ценности ВСЕХ состояний при следовании данной стратегии
    """
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    V_his = [V]
    i = 0
    while True:
        V_temp = torch.zeros(n_state)
        i += 1

        # state - индекс!
        for state, actions in enumerate(policy):

            # action - индекс!
            for action, action_prob in enumerate(actions):
                V_temp[state] += action_prob * (rewards[state] + gamma * torch.dot(trans_matrix[state, action], V))
        
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        V_his.append(V)

        if max_delta <= treshold:
            break

    return V, V_his


V, V_his = policy_eval(policy_optimal, T, R, gamma, treshold)
print('Функция ценности при оптимальной стратегии: \n{}'.format(V))
plot_history(V_his, gamma)

gamma = 0.99
V, V_his = policy_eval(policy_optimal, T, R, gamma, treshold)
print('Функция ценности при оптимальной стратегии: \n{}'.format(V))
plot_history(V_his, gamma)

gamma = 0.001
V, V_his = policy_eval(policy_optimal, T, R, gamma, treshold)
print('Функция ценности при оптимальной стратегии: \n{}'.format(V))
plot_history(V_his, gamma)

# policy_random = torch.tensor([[0.5, 0.5],
#                               [0.5, 0.5],
#                               [0.5, 0.5]])

# V, V_his = policy_eval(policy_random, T, R, gamma, treshold)
# print('Функция ценности при случайной стратегии: \n{}'.format(V))

# policy_play = torch.tensor([[0.0, 1.0],
#                             [1.0, 0.0],
#                             [0.1, 0.9]])

# R_play = torch.tensor([0.4, 0.4, 0.2])
# V, V_his = policy_eval(policy_play, T, R_play, gamma, treshold)
# print('Функция ценности при стратегии поиграть: \n{}'.format(V))