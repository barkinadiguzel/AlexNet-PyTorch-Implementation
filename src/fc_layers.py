import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedLayers(nn.Module):
    def __init__(self, dropout=True):
        super(FullyConnectedLayers, self).__init__()
        self.fc6 = nn.Linear(256*6*6, 4096)  # Conv5's output 256x6x6, to be flattened
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)      # 1000 class

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc6(x))
        if self.dropout:
            x = self.drop(x)
        x = F.relu(self.fc7(x))
        if self.dropout:
            x = self.drop(x)
        x = self.fc8(x)
        return x
