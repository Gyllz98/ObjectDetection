import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class PotholeCNN(nn.Module):
    def __init__(self):
        super(PotholeCNN, self).__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=32),
            F.relu(),
        ) 

        # Define the network layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 1)  # Binary classification output

        self.dropout = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.bn_fc = nn.BatchNorm1d(num_features=512)
        
        # Apply weight initialization
        self._initialize_weights()
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers

        x = torch.relu(self.bn_fc(self.fc1(x)))

        x = self.dropout(x)
        
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class PretrainedPothole(nn.Module):
    def __init__(self, backbone = "VGG16", pretrained=True, freeze_backbone = False):
        super(PretrainedPothole, self).__init__()
        match backbone:
            case "ResNet101":
                # Load the ResNet-101 model
                if pretrained:
                    self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
                else:
                    self.model = models.resnet101(weights=None)

                # Replace the fully connected layer (for ImageNet) with one for binary classification
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, 1)  # Output 1 logit for binary classification
                # Remove the original fully connected layer
                self.model.fc = nn.Identity()

                # Add custom layers with dropout
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 1)
                )

                if freeze_backbone:
                    for param in self.model.parameters():
                        param.requires_grad = False
            case "VGG16":
                if pretrained:
                    self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                else:
                    self.model = models.vgg16(weights=None)

                last_layer = self.model.classifier[-1]

                self.model.classifier[-1] = nn.Linear(last_layer.in_features,1)

                if freeze_backbone:
                    for param in self.model.features.parameters():
                        param.requires_grad = False
                    for param in self.model.classifier[-3].parameters():
                        param.requires_grad = True
            case None:
                print("No model was selected.")

    def forward(self, x):
        x = self.model(x)
        # x = self.classifier(x)
        return x