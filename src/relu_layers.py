import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x):
        return self.relu(x)
