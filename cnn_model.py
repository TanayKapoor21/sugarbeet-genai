import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, num_channels=96, num_classes=4):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(num_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.6),

            nn.Linear(32, num_classes)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x