"""
Simple CNN Baseline
===================
Lightweight 2-layer CNN for GOFLOW velocity field prediction.

A minimal baseline model to compare against more complex architectures.
Uses only same-padding convolutions, preserving spatial dimensions.

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import torch.nn as nn


class TwoLayerCNN(nn.Module):
    """
    Simple 2-layer CNN baseline for velocity prediction.

    Architecture:
        - Optional input batch normalization
        - Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        - All convolutions use same padding (output size = input size)

    This lightweight model serves as a baseline to compare against
    U-Net and other more complex architectures.

    Args:
        n_channels: Number of input channels (default 3 for SST frames)
        no: Number of output channels (default 2 for U,V velocity)
        hidden: Hidden channel count (default 64)
        kernel_size: Convolution kernel size (default 7, try 5 or 15)
        inpNorm: If True, apply batch normalization to input
    """

    def __init__(self, n_channels=3, no=2, hidden=64, kernel_size=7, inpNorm=True):
        super().__init__()
        padding = kernel_size // 2
        self.inpNorm = inpNorm

        if inpNorm:
            self.bn0 = nn.BatchNorm2d(n_channels)

        self.conv1 = nn.Conv2d(n_channels, hidden, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, no, kernel_size, padding=padding)

    def forward(self, x):
        if self.inpNorm:
            out = self.bn0(x)
            out = self.conv2(self.relu(self.bn1(self.conv1(out))))
        else:
            out = self.conv2(self.relu(self.bn1(self.conv1(x))))
        return self.conv3(self.relu(self.bn2(out)))
