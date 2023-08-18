from torch import nn


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes: int, num_frames: int, batch_size: int):

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.batch_size = batch_size

        super(Simple3DCNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 4 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
