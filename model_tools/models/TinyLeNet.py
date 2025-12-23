import torch.nn as nn
import torch.nn.functional as F


class TinyLeNet(nn.Module):
    def __init__(self):
        super(TinyLeNet, self).__init__()
        # C1: 1通道输入, 6通道输出, 5x5核
        self.conv1 = nn.Conv2d(1, 6, 5)
        # C3: 6通道输入, 16通道输出, 5x5核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # S2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # S4
        x = x.view(-1, 16 * 4 * 4)                 # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
