import gym
import torch
from torch.autograd import Variable
import random

class DQN():
    
    def __init__(self, n_state, n_action, n_hidden = 50, lr = 0.05) -> None:
        self.criterion = torch.nn.MSELoss()
        self.li = torch.nn.Linear(n_state, n_hidden)
        self.lo = torch.nn.Linear(n_hidden, n_action)
        self.model = torch.nn.Sequential(
            self.li,
            torch.nn.ReLU(),
            self.lo
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

    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []

            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)

            self.update(states, td_targets)

    def save(self):
        torch.save(self.model.state_dict(), 'model.pt')

    def load(self):
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()

    def print(self):
        print('li:')
        print(self.li.weight.data.numpy())
        print('lo:')
        print(self.lo.weight.data.numpy())
