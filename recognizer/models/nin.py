import torch
import torch.nn as nn


class NiNVideoClassifier(nn.Module):
    def __init__(self, num_classes: int, num_frames: int, batch_size: int):
        super(NiNVideoClassifier, self).__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.batch_size = batch_size

        self.nin_block1 = self._make_nin_block(3, 48, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.nin_block2 = self._make_nin_block(48, 32, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.nin_block3 = self._make_nin_block(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.nin_block4 = self._make_nin_block(64, num_classes, kernel_size=(1, 1, 1), stride=1, padding=0)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

    def _make_nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.nin_block1(x)
        x = self.nin_block2(x)
        x = self.maxpool(x)

        x = self.nin_block3(x)
        x = self.nin_block4(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x
