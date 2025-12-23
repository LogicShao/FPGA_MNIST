import torch.nn as nn

HIDDEN_SIZE = 32  # 隐层节点数，FPGA资源有限，32或16比较合适


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        # Layer 1: 784 -> 32
        self.fc1 = nn.Linear(28 * 28, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        # Layer 2: 32 -> 10
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
