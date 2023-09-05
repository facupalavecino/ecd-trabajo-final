from torch import nn


class Molchanov3DCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        batch_size: int
    ):

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.batch_size = batch_size

        super(Molchanov3DCNN, self).__init__()

        self.conv_layer_1 = nn.Sequential(
            # Input shape <C, D, H, W> (3, 16, 216, 384)
            nn.Conv3d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=0),
            # Output shape <C, D, H, W> (3, 12, 212, 380)
            nn.ReLU(inplace=True),
            # Output ReLU shape <C, D, H, W> (3, 12, 212, 380)
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Output pooling shape <C, D, H, W> (3, 6, 106, 190) divide todo por el stride
        )

        self.conv_layer_2 = nn.Sequential(
            # Input shape <C, D, H, W> (3, 6, 106, 190)
            nn.Conv3d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=0),
            # Output shape <C, D, H, W> (5, 4, 104, 188)
            nn.ReLU(inplace=True),
            # Output ReLU shape <C, D, H, W> (5, 4, 104, 188)
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Output pooling shape <C, D, H, W> (5, 2, 52, 94)
        )

        self.fc_layer = nn.Sequential( 
            nn.Linear(5 * 2 * 52 * 94, 100),
            # <C * D * H * W> (5 * 2 * 25 * 46 si la entrada es 10% 16 frame)
            # <C * D * H * W> (5 * 2 * 52 * 94 si la entrada es 20% 16 frame)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer_1(x)

        x = self.conv_layer_2(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
