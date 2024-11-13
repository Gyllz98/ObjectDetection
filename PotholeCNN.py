import torch
from torch import nn

class PotholeCNN(nn.Module):
    def __init__(self):
        super(PotholeCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 512)  # Adjust input size based on the spatial dimensions after conv layers
        self.fc2 = nn.Linear(512, 1)  # Binary classification (pothole or not)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Use Sigmoid for binary output

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x

