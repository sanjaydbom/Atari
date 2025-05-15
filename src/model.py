import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (8,8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4,4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), 1)
        self.fc1 = nn.Linear(3136,256)
        self.output = nn.Linear(256,4)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.output(x)
        return x