import torch
import torch.nn as nn
from conv_layers import Conv1, Conv2, Conv3, Conv4, Conv5
from relu_layers import ReLU
from pool_layers import MaxPool
from normalization_layers import LRN
from fc_layers import FC6, FC7, FC8

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = Conv1()
        self.conv2 = Conv2()
        self.conv3 = Conv3()
        self.conv4 = Conv4()
        self.conv5 = Conv5()
        
        # ReLU
        self.relu = ReLU()
        
        # Pooling
        self.pool = MaxPool()
        
        # Normalization
        self.lrn = LRN()
        
        # Fully-connected layers
        self.fc6 = FC6()
        self.fc7 = FC7()
        self.fc8 = FC8(num_classes=num_classes)
        
    def forward(self, x):
        # Conv1 -> ReLU -> LRN -> Pool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pool(x)
        
        # Conv2 -> ReLU -> LRN -> Pool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pool(x)
        
        # Conv3 -> ReLU
        x = self.conv3(x)
        x = self.relu(x)
        
        # Conv4 -> ReLU
        x = self.conv4(x)
        x = self.relu(x)
        
        # Conv5 -> ReLU -> Pool
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten for fully-connected layers
        x = x.view(x.size(0), -1)
        
        # Fully-connected layers
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        
        return x
