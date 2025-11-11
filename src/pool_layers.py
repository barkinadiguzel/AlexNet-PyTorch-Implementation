import torch
import torch.nn as nn

# Max pooling layer with overlapping
# Kernel size = 3, stride = 2 (overlapping)
class MaxPoolLayer(nn.Module):
    def __init__(self):
        super(MaxPoolLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  

    def forward(self, x):
        return self.pool(x)
