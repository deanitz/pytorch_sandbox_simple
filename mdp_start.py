import torch

T = torch.tensor([[[0.8, 0.1, 0.1],
                   [0.1, 0.6, 0.3]],
                  [[0.7, 0.2, 0.1],
                   [0.1, 0.8, 0.1]],
                  [[0.6, 0.2, 0.2],
                   [0.1, 0.4, 0.5]]])

R = torch.tensor([1., 0., -1.])
gamma = 0.5
action = 0

def calc_value_matrix_inversion(gamma, trans_matrix, rewards):
    inv = torch.inverse(torch.eye(rewards.shape[0]) - gamma * trans_matrix)
    V = torch.mm(inv, rewards.reshape(-1, 1))
    return V

trans_matrix = T[:, action]
V = calc_value_matrix_inversion(gamma, trans_matrix, R)
print('Функция ценности при оптимальной стратегии: \n{}'.format(V))