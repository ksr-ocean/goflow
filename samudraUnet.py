"""
Samudra U-Net Architecture
===========================
ConvNeXt-inspired U-Net variant for GOFLOW velocity field prediction.
Dheeshjith et al "Samudra: An AI global ocean emulator for climate", GRL, 52.10 (2025): e2024GL114318
..."inspired" because a lot of the things that make ConvNeXts work are actually not present.

Uses modern ConvNeXt-style blocks with:
- Depth-wise separable-like convolutions
- GELU activations
- Batch normalization
- Dilation for increased receptive field
- Skip connections in encoder-decoder

Two variants available via padding_mode:
- 'zeros': Standard zero padding (samudra0)
- 'reflect': Reflection padding for better boundary handling (samudraR)

"""

import torch
import torch.nn as nn


# =============================================================================
# Building Blocks
# =============================================================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style convolutional block.

    Uses inverted bottleneck structure:
        input -> expand (1x1) -> depthwise conv -> contract (1x1) -> output

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Convolution kernel size
        dilation: Dilation factor for increased receptive field
        upscale_factor: Channel expansion factor in hidden layer
        padding_mode: 'zeros' or 'reflect' for boundary handling
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilation=1, upscale_factor=4, padding_mode='zeros'):
        super().__init__()
        
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        hidden = in_channels * upscale_factor
        
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size, 
                               dilation=dilation, padding=padding, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size, 
                               dilation=dilation, padding=padding, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(hidden, out_channels, 1)
    
    def forward(self, x):
        skip = self.skip(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return skip + x


# =============================================================================
# Main Model
# =============================================================================

class SamudraUNet(nn.Module):
    """
    Samudra U-Net: ConvNeXt-inspired encoder-decoder for velocity prediction.

    Architecture:
        - Input projection (1x1 conv)
        - 2 encoder stages with ConvNeXt blocks + AvgPool
        - Bottleneck ConvNeXt block
        - 2 decoder stages with ConvNeXt blocks + bilinear upsample
        - Skip connections from encoder to decoder
        - Output projection (1x1 conv)

    Args:
        n_channels: Number of input channels (default 3 for SST frames)
        no: Number of output channels (default 2 for U,V velocity)
        Nbase: Base channel multiplier (channels = Nbase * [4, 16, 32])
        dilation: List of dilation factors for each encoder level
        padding_mode: 'zeros' or 'reflect' for boundary handling
        inpNorm: If True, apply batch normalization to input
    """

    def __init__(self, n_channels=3, no=2,
                 Nbase=16,
                 dilation=[1, 2, 4],
                 padding_mode='zeros',
                 inpNorm=True):
        super().__init__()
        self.inpNorm = inpNorm
        ch_width=[Nbase*4, Nbase*16, Nbase*32]
        if inpNorm:
            self.bn0 = nn.BatchNorm2d(n_channels)
        self.num_steps = len(ch_width) - 1
        
        # Initial projection from input channels to first width
        self.in_proj = nn.Conv2d(n_channels, ch_width[0], 1)
        
        layers = []
        
        # Encoder
        for i in range(len(ch_width) - 1):
            layers.append(ConvNeXtBlock(ch_width[i], ch_width[i+1], 
                                       dilation=dilation[i], padding_mode=padding_mode))
            layers.append(nn.AvgPool2d(2))
        
        # Bottleneck
        layers.append(ConvNeXtBlock(ch_width[-1], ch_width[-1], 
                                   dilation=dilation[-1], padding_mode=padding_mode))
        
        # Decoder
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        for i in range(len(ch_width) - 2, -1, -1):
            layers.append(ConvNeXtBlock(ch_width[i+1], ch_width[i], 
                                       dilation=dilation[i], padding_mode=padding_mode))
            if i > 0:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        # Final projection from last width to output channels
        layers.append(nn.Conv2d(ch_width[0], no, 1))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        if self.inpNorm:
            x = self.bn0(x)
        x = self.in_proj(x)  # Add this!
        
        temp = []
        count = 0
        
        for layer in self.layers:
            x = layer(x)
            if count < self.num_steps and isinstance(layer, ConvNeXtBlock):
                temp.append(x)
                count += 1
            elif count >= self.num_steps and isinstance(layer, nn.Upsample):
                x = x + temp[2 * self.num_steps - count - 1]
                count += 1
        return x
