import torch
import torch.nn as nn

# Conv1: 224x224x3 -> 55x55x96 (stride=4, kernel=11)
class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return x


# Conv2: 27x27x96 -> 27x27x256
class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)  # padding 2 to keep spatial size
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        return x


# Conv3: 13x13x256 -> 13x13x384
class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# Conv4: 13x13x384 -> 13x13x384
class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        self.conv = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# Conv5: 13x13x384 -> 13x13x256 -> 6x6x256 after pooling
class Conv5(nn.Module):
    def __init__(self):
        super(Conv5, self).__init__()
        self.conv = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
