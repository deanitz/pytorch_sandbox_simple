import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, n_channel, n_action) -> None:
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels= n_channel,
            out_channels= 32,
            kernel_size=8,
            stride=4
            )
        self.conv2 = nn.Conv2d(
            in_channels= 32,
            out_channels= 64,
            kernel_size=4,
            stride=2
            )
        self.conv3 = nn.Conv2d(
            in_channels= 64,
            out_channels= 64,
            kernel_size=3,
            stride=1
            )
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, n_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc(x))
        output = self.out(x)
        return output