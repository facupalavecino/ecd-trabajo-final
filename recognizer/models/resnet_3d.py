import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Basic3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class Simple3DResNet(nn.Module):
    def __init__(self, num_classes=64):
        super(Simple3DResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = Basic3DBlock(16, 16, stride=1)
        self.layer2 = Basic3DBlock(16, 32, stride=2)
        self.layer3 = Basic3DBlock(32, 64, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Input shape (3, 16, 216, 384)
        x = self.conv1(x)  # shape becomes (16, 8, 108, 192)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)  # shape becomes (16, 8, 108, 192)
        x = self.layer2(x)  # shape becomes (32, 4, 54, 96)
        x = self.layer3(x)  # shape becomes (64, 2, 27, 48)
        
        x = self.avgpool(x)  # shape becomes (64, 1, 1, 1)
        x = torch.flatten(x, 1)  # flatten the tensor
        x = self.fc(x)  # shape becomes (num_classes)
        
        return x
