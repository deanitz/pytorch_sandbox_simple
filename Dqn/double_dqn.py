import gym
import torch
from torch.autograd import Variable
import random
import copy

class DoubleDQN():
    
    def __init__(self, n_state, n_action, n_hidden = 50, lr = 0.05) -> None:
        
        
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))

        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0")

        self.criterion = torch.nn.MSELoss()
        self.li = torch.nn.Linear(n_state, n_hidden)
        self.lo = torch.nn.Linear(n_hidden, n_action)
        self.model = (torch.nn.Sequential(
            self.li,
            torch.nn.ReLU(),
            self.lo
        )).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.model_target = copy.deepcopy(self.model).to(self.device)

    def update(self, state, target):
        prediction = self.model(torch.Tensor(state).to(self.device))
        loss = self.criterion(prediction, Variable(torch.Tensor(target).to(self.device)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state).to(self.device))
    
    def predict_target(self, state):
        with torch.no_grad():
            return self.model_target(torch.Tensor(state).to(self.device))

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

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
                    q_values_next = self.predict_target(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)

            self.update(states, td_targets)

    def save(self):
        torch.save(self.model.state_dict(), 'double_dqn_model.pt')

    def load(self):
        self.model.load_state_dict(torch.load('double_dqn_model_trained.pt'))
        self.model.eval()

    def print(self):
        print('li:')
        print(self.li.weight.data.numpy())
        print('lo:')
        print(self.lo.weight.data.numpy())
