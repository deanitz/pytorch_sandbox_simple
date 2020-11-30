import os
import torch
from torch.autograd import Variable
import random
from cnn_model import CNNModel

class CNN_DQN():
    
    def __init__(self, n_channel, n_action, gpu=False, lr = 0.05) -> None:

        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0") if gpu else torch.device("cpu")

        self.criterion = torch.nn.MSELoss()
        self.model = CNNModel(n_channel, n_action).to(self.device)
        self.model_target = CNNModel(n_channel, n_action).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

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
                states.append(state.tolist()[0])
                q_values = self.predict(state).tolist()[0]
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict_target(next_state).detach()
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)

            self.update(states, td_targets)

    def save(self):
        torch.save(self.model.state_dict(), 'cnn_dqn_model.pt')

    def load(self):
        if os.path.isfile('cnn_dqn_model.pt'):
            self.model.load_state_dict(torch.load('cnn_dqn_model.pt'))
            self.model.eval()
            print('Loaded model.')
        else:
            print('No model is found. New model initialized.')

    def gen_epsilon_greedy_policy(self, epsilon, n_action):
        def policy_function(state):
            if random.random() < epsilon:
                return random.randint(0, n_action - 1)
            else:
                q_values = self.predict(state)
                return torch.argmax(q_values).item()
        return policy_function

