from torch import nn


class Alt3DCNN_v2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        batch_size: int
    ):
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.batch_size = batch_size

        super(Alt3DCNN_v2, self).__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(4 * 4 * 54 * 96, 200),  # Increased from 100 to 200
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 100),  # New layer
            nn.ReLU(inplace=True),  # New activation
            nn.Dropout(p=0.5),  # Additional Dropout
            nn.Linear(100, num_classes)  # Final layer remains the same
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x
