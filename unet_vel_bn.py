"""
U-Net Model Architecture
=========================
U-Net implementation for GOFLOW velocity field prediction.

Adapted from: https://github.com/milesial/Pytorch-UNet

Modifications:
1. Nbase parameter to scale model capacity without changing topology
2. Optional input batch normalization layer

Also includes:
- VelocityField: Stream function formulation for velocity prediction
- VortDivField: Direct vorticity/divergence prediction wrapper
- Discriminator/PatchDiscriminator: For potential GAN training

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import torch
import torch.nn as nn
from unet_parts_t import Down, DoubleConv, MixPool2d, Up, TemporalEncoding, OutConv


# =============================================================================
# Main U-Net Model
# =============================================================================

class UNet(nn.Module):
    """
    U-Net encoder-decoder architecture for image-to-image prediction.

    Architecture:
        - 4 encoder blocks with strided convolutions (downsampling)
        - Bottleneck layer
        - 4 decoder blocks with skip connections (upsampling)
        - 1x1 output convolution

    Args:
        n_channels: Number of input channels (e.g., 3 for 3-frame SST input)
        no: Number of output channels (e.g., 2 for U,V velocity)
        bilinear: If True, use bilinear upsampling; if False, use transposed conv
        Nbase: Base channel count (model size scales with this)
        inpNorm: If True, apply batch normalization to input
    """

    def __init__(self, n_channels, no, bilinear=False, Nbase=16, inpNorm=False):
        super().__init__()
        #self.tempEmb = TemporalEncoding(n_emb = n_emb)
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inpNorm = inpNorm
        if inpNorm:
            self.bn = nn.BatchNorm2d(n_channels)
        #self.inc = DoubleConvTime(n_channels, Nbase, n_emb = n_emb)
        self.inc = (DoubleConv(n_channels, Nbase, stride = 1))
        self.down1 = (Down(Nbase, Nbase*2))
        self.down2 = (Down(Nbase*2, Nbase*4))
        self.down3 = (Down(Nbase*4, Nbase*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(Nbase*8, Nbase*16 // factor))
        self.up1 = (Up(Nbase*16, Nbase*8 // factor, bilinear))
        self.up2 = (Up(Nbase*8, Nbase*4 // factor, bilinear))
        self.up3 = (Up(Nbase*4, Nbase*2 // factor, bilinear))
        self.up4 = (Up(Nbase*2, Nbase, bilinear))
        self.outc = self.conv = nn.Conv2d(Nbase, no, kernel_size=1)

    def forward(self, x):
        if self.inpNorm:
            x1 = self.inc(self.bn(x))
        else:
            x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        #return x
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# =============================================================================
# Gradient Operators (for physics-based models)
# =============================================================================

def gradient_x(y: torch.Tensor, pm: float = 1) -> torch.Tensor:
    """Compute x-derivative using centered finite difference."""
    kernel = pm * torch.Tensor([-1., 0., 1.]).view(1, 1, 1, 3) / 2
    kernel = kernel.to(y.device)
    return nn.functional.conv2d(y, kernel, padding=(0, 1))


def gradient_y(y: torch.Tensor, pn: float = 1) -> torch.Tensor:
    """Compute y-derivative using centered finite difference."""
    kernel = pn * torch.Tensor([-1., 0., 1.]).view(1, 1, 3, 1) / 2
    kernel = kernel.to(y.device)
    return nn.functional.conv2d(y, kernel, padding=(1, 0))


# =============================================================================
# Physics-Based Velocity Models
# =============================================================================

class VelocityField(nn.Module):
    """
    Velocity field from stream function and velocity potential.

    Predicts (psi, phi) and computes velocity as:
        U = d(phi)/dx + d(psi)/dy
        V = d(phi)/dy - d(psi)/dx

    This formulation naturally satisfies certain physical constraints.
    """

    def __init__(self, n_channels, bilinear=False, Nbase=16, inpBNFlag=False, pm=1., pn=1.):
        super(VelocityField, self).__init__()
        self.unet = UNet(n_channels, bilinear, Nbase, inpBNFlag)

    def forward(self, x):
        psi_phi = self.unet(x)

        # split psi and phi
        psi, phi = psi_phi.split(1, dim=1)

        # Compute gradients
        phi_x = gradient_x(phi, pm)
        phi_y = gradient_y(phi, pn)
        psi_x = gradient_x(psi, pm)
        psi_y = gradient_y(psi, pn)

        # Compute velocity fields
        U = phi_x + psi_y
        V = phi_y - psi_x
        return torch.cat([U, V], dim = 1)

def vortDiv(uv: torch.Tensor, pm: float = 1, pn: float = 1, f: float = 1) -> tuple:
    """
    Compute vorticity and divergence from velocity field.

    Args:
        uv: Velocity tensor (B, 2, H, W) with U in channel 0, V in channel 1
        pm: X grid metric
        pn: Y grid metric
        f: Coriolis parameter for normalization

    Returns:
        Tuple of (vorticity, divergence)
    """
    u, v = uv.split(1, dim=1)

    u_x = gradient_x(u, pm)
    u_y = gradient_y(u, pn)
    v_x = gradient_x(v, pm)
    v_y = gradient_y(v, pn)

    div = (u_x + v_y) / f
    vort = (v_x - u_y) / f
    return (vort, div)


class VortDivField(nn.Module):
    """
    U-Net wrapper that also computes vorticity and divergence.

    Predicts velocity field and returns derived vorticity/divergence fields.
    """

    def __init__(self, n_channels, no, bilinear=False, Nbase=16, inpBNFlag=False, pm=1., pn=1., f=1):
        super(VortDivField, self).__init__()
        self.unet = UNet(n_channels, no, bilinear, Nbase, inpBNFlag)
        self.pm, self.pn, self.f = pm, pn, f
    def forward(self, x):
        uv = self.unet(x)
        vort_div =  vortDiv(uv, self.pm, self.pn, self.f)
        # split psi and phi
        return vort_div

# =============================================================================
# GAN Discriminators
# =============================================================================

class Discriminator(nn.Module):
    """
    Global discriminator for GAN training.

    Uses strided convolutions for downsampling with spectral normalization
    for stable training. Outputs a single scalar per sample.

    Args:
        input_channels: Number of input channels (e.g., 2 for velocity field)
        base_filters: Base filter count (doubles at each layer)
    """

    def __init__(self, input_channels, base_filters=64):
        super(Discriminator, self).__init__()

        self.down1 = self.discriminator_block(input_channels, base_filters, normalization=False)
        self.down2 = self.discriminator_block(base_filters, base_filters * 2)
        self.down3 = self.discriminator_block(base_filters * 2, base_filters * 4)
        self.down4 = self.discriminator_block(base_filters * 4, base_filters * 8, stride=1)

        # Final layer
        #self.final = nn.Conv2d(base_filters * 8, 1, kernel_size=4, padding=1, stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_filters * 8, 1)

        # Apply spectral normalization to each convolutional layer
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.utils.spectral_norm(module)
    def discriminator_block(self, in_filters, out_filters, normalization=True, stride=2):
        """Returns layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
        if normalization:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return torch.sigmoid(x)

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for local texture discrimination.

    Outputs a spatial map of real/fake predictions rather than a single
    scalar, encouraging the generator to produce locally realistic textures.

    Concatenates SST input with velocity output for conditional discrimination.

    Args:
        input_channels: Total input channels (SST + velocity, default 5 = 3 + 2)
        ndf: Number of discriminator filters in first layer
    """

    def __init__(self, input_channels=5, ndf=64):
        super().__init__()

        self.layers = nn.Sequential(
            # input is (input_channels) x 256 x 256
            nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            # state size: (ndf) x 128 x 128

            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # state size: (ndf*2) x 64 x 64

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # state size: (ndf*4) x 32 x 32

            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # state size: (ndf*8) x 16 x 16

            nn.Conv2d(ndf * 8, 1, 4, stride=2, padding=1)
            # output size: 1 x 8 x 8
        )

    def forward(self, sst, velocity):
        x = torch.cat([sst, velocity], dim=1)  # Concatenate along channel dimension
        return self.layers(x)
