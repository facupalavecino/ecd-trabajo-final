import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class ResNet3DClassifier(nn.Module):
    def __init__(self, num_classes=64, pretrained=True):
        super(ResNet3DClassifier, self).__init__()

        # Initialize a pretrained 3D ResNet-18 model
        self.base_model = r3d_18(pretrained=pretrained)

        # Remove the last fully connected layer to use the model as a feature extractor
        modules = list(self.base_model.children())[:-1]
        self.base_model = nn.Sequential(*modules)

        # Add a new fully connected layer to match the number of classes in your sign language dataset
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        
        # Reshape the tensor before feeding it into the fully connected layer
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x
