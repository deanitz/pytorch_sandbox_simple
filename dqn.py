import gym
import torch
from torch.autograd import Variable
import random

class DQN():
    
    def __init__(self, n_state, n_action, n_hidden = 50, lr = 0.05) -> None:
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, target):
        prediction = self.model(torch.Tensor(state))
        loss = self.criterion(prediction, Variable(torch.Tensor(target)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state))
