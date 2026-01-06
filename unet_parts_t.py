"""
U-Net Building Blocks
=====================
Modular components for constructing U-Net architectures.

Adapted from: https://github.com/milesial/Pytorch-UNet

Components:
- MixPool2d: Combined average + max pooling
- DoubleConv: Standard (conv -> BN -> activation) x 2
- DoubleConvLN: LayerNorm variant of DoubleConv
- DoubleConvTime: Time-conditioned DoubleConv (for diffusion models)
- Down: Downsampling block
- Up: Upsampling block with skip connections
- OutConv: 1x1 output convolution

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Pooling Layers
# =============================================================================

class MixPool2d(nn.Module):
    """
    Mixed pooling: concatenates average and max pooling, then reduces channels.

    This combines the benefits of both pooling strategies:
    - Average pooling preserves smooth features
    - Max pooling preserves sharp features/edges
    """

    def __init__(self, planes):
        super(MixPool2d, self).__init__()
        self.avpool = nn.AvgPool2d(2)
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(2*planes, planes, 1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.avpool(x)
        x2 = self.maxpool(x)
        out = torch.cat([x1, x2],1)
        return self.conv1(out)

# =============================================================================
# Temporal Encoding (for time-conditioned models)
# =============================================================================

class TemporalEncoding(nn.Module):
    """
    Sinusoidal temporal encoding for time-conditioned models.

    Converts scalar time values to learned embeddings using sin activation.
    """

    def __init__(self, n_emb=100):
        super().__init__()
        self.inp = nn.Linear(1, 2 * n_emb)
        self.act = torch.sin
        self.out = nn.Linear(2 * n_emb, n_emb)

    def forward(self, t):
        return self.out(self.act(self.inp(t)))


# =============================================================================
# Convolutional Blocks
# =============================================================================

class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv -> BN -> SiLU) x 2

    Standard U-Net building block. First conv can optionally downsample
    via stride parameter.

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        mid_channels: Intermediate channel count (defaults to out_channels)
        stride: Stride for first convolution (2 for downsampling, 1 for same size)
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, stride=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, stride = stride),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class DoubleConvLN(nn.Module):
    """
    Double convolution with Layer Normalization instead of Batch Norm.

    Uses manually implemented spatial LayerNorm with learnable scale/shift.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        #self.short1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.gamma1 = nn.Parameter(torch.ones(1, mid_channels, 1, 1))
        self.beta1 = nn.Parameter(torch.zeros(1, mid_channels, 1, 1))

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        #self.short2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
        self.gamma2 = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.beta2 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def layer_norm(self, x, gamma, beta):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        return gamma * (x - mean) / (std + 1e-5) + beta

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer_norm(x, self.gamma1, self.beta1)
        x = nn.SiLU()(x)
        
        x = self.conv2(x)
        x = self.layer_norm(x, self.gamma2, self.beta2)
        x = nn.SiLU()(x)
        
        return x

class DoubleConvTime(nn.Module):
    """
    Time-conditioned double convolution for diffusion models.

    Applies time embedding via multiplicative modulation after each conv block.

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        mid_channels: Intermediate channel count (defaults to out_channels)
        n_emb: Time embedding dimension
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, n_emb=100):
        super().__init__()
        #super(DoubleConvTime, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        #BxC
        self.time1 = nn.Linear(n_emb, mid_channels)
        self.time2 = nn.Linear(n_emb, out_channels)
        #BxCxHxW
        self.c1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1= nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)
        
        self.c2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, timeEmb):
        
        out = self.act(self.bn1(self.c1(x)))
        t1 = self.time1(timeEmb)
        #print('Inside DoubleConv')
        #print(f'out:{out.shape}, t1: {t1.shape}')
        out = out + out * t1[:, :,None, None]
        #out = out + out * t1.expand_as(out)
        
        out = self.act(self.bn2(self.c2(out)))
        t2 = self.time2(timeEmb)
        #print(f'out:{out.shape},t2: {t2.shape}')
        out = out + out * t2[:, :,None, None]

        return out


# =============================================================================
# Encoder/Decoder Blocks
# =============================================================================

class DownTime(nn.Module):
    """Time-conditioned downsampling block with mixed pooling."""

    def __init__(self, in_channels, out_channels, n_emb=100):
        super(DownTime, self).__init__()
        self.pool = MixPool2d(in_channels)
        self.conv = DoubleConvTime(in_channels, out_channels, n_emb=n_emb)

    def forward(self, x, timeEmb):
        out = self.pool(x)
        return self.conv(x, timeEmb)


class Down(nn.Module):
    """
    Downsampling block: strided convolution (via DoubleConv).

    Note: Downsampling is done via stride in DoubleConv, not pooling.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        #self.maxpool_conv = nn.Sequential(
            #MixPool2d(in_channels),
            #nn.MaxPool2d(2),
            #DoubleConv(in_channels, out_channels)
        #)

    def forward(self, x):
        #return self.maxpool_conv(x)
        return self.conv(x)


class Up(nn.Module):
    """
    Upsampling block with skip connection.

    Upsamples the input, concatenates with skip connection from encoder,
    then applies double convolution.

    Args:
        in_channels: Input channels (after concatenation with skip)
        out_channels: Output channel count
        bilinear: If True, use bilinear upsampling; if False, use transposed conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, stride=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# =============================================================================
# Output Layer
# =============================================================================

class OutConv(nn.Module):
    """1x1 output convolution to map features to desired output channels."""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
