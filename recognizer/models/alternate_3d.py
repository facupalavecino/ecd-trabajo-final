from torch import nn


class Alt3DCNN_v2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
       batch_size: int,
       input_shape
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.batch_size = batch_size

        super(Alt3DCNN_v2, self).__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1),
            nn.BatchNorm3d(num_features=4),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(num_features=8),
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(num_features=12),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(12 * 9 * 28 * 49, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.25),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x
