import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Global Perceptron
# -----------------------------
class GlobalPerceptron(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(in_channels, out_channels),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(out_channels, out_channels)

        )

    def forward(self, x):

        b, c, _, _ = x.shape

        g = self.pool(x).view(b, c)

        g = self.fc(g)

        g = g.unsqueeze(-1).unsqueeze(-1)

        return g


# -----------------------------
# Partition Perceptron
# -----------------------------
class PartitionPerceptron(nn.Module):

    def __init__(self, in_channels, out_channels, groups=4):

        super().__init__()

        self.block = nn.Sequential(

            nn.GroupNorm(4, in_channels),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=groups
            ),

            nn.ReLU(),

            nn.Dropout2d(0.2)

        )

    def forward(self, x):

        return self.block(x)


# -----------------------------
# Local Perceptron
# -----------------------------
class LocalPerceptron(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.dilated_convs = nn.ModuleList([

            nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=d,
                    dilation=d
                ),

                nn.GroupNorm(4, out_channels),

                nn.ReLU()

            )

            for d in [1, 2, 3]

        ])

    def forward(self, x):

        outputs = [conv(x) for conv in self.dilated_convs]

        return sum(outputs)


# -----------------------------
# DMLP Block
# -----------------------------
class DMLPBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.global_perc = GlobalPerceptron(in_channels, out_channels)

        self.partition_perc = PartitionPerceptron(in_channels, out_channels)

        self.local_perc = LocalPerceptron(in_channels, out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.norm = nn.GroupNorm(4, out_channels)

        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):

        g = self.global_perc(x)

        p = self.partition_perc(x)

        l = self.local_perc(x)

        fusion = p + l

        fusion = fusion + self.shortcut(x)

        fusion = fusion + g

        fusion = self.norm(fusion)

        fusion = F.relu(fusion)

        fusion = self.dropout(fusion)

        return fusion


# -----------------------------
# Main DMLPFFN Network
# -----------------------------
class DMLPFFN(nn.Module):

    def __init__(self, in_channels=96, num_classes=4):

        super().__init__()

        # Low-level branch
        self.low_branch = nn.Sequential(

            nn.Conv2d(in_channels, 16, 3, padding=1),

            nn.GroupNorm(4, 16),

            nn.ReLU(),

            DMLPBlock(16, 16),

            nn.Conv2d(16, 64, kernel_size=1)

        )

        # Mid-level branch
        self.mid_branch = nn.Sequential(

            nn.Conv2d(in_channels, 32, 5, padding=2),

            nn.GroupNorm(4, 32),

            nn.ReLU(),

            DMLPBlock(32, 32),

            nn.Conv2d(32, 64, kernel_size=1)

        )

        # High-level branch
        self.high_branch = nn.Sequential(

            DMLPBlock(in_channels, 64),

            nn.Conv2d(64, 64, kernel_size=1)

        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.6)

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):

        low = self.low_branch(x)

        mid = self.mid_branch(x)

        high = self.high_branch(x)

        fused = low + mid + high

        pooled = self.gap(fused).flatten(1)

        pooled = nn.functional.dropout(
        pooled,
        p=0.6,
        training=self.training
)

        out = self.classifier(pooled)

        return out