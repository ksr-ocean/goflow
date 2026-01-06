"""
Spectral Loss Functions
========================
Loss functions operating in the frequency domain for GOFLOW training.

Includes:
- spectral_loss: MSE in log-space of 2D FFT kinetic energy spectrum
- spectral_loss_mirror: Mirror-padded spectral loss to reduce boundary effects
- spectral_loss_directional: Separate x/y directional spectral losses

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.fft import rfft, irfft, rfftn, irfftn


# =============================================================================
# FFT Utilities
# =============================================================================

def rft(u: torch.Tensor) -> torch.Tensor:
    """2D real FFT along last two dimensions."""
    return rfftn(u, dim=[-2, -1])


def rft1(u: torch.Tensor) -> torch.Tensor:
    """1D real FFT along last dimension (x-direction)."""
    return rfftn(u, dim=[-1])


def rft2(u: torch.Tensor) -> torch.Tensor:
    """1D real FFT along second-to-last dimension (y-direction)."""
    return rfftn(u, dim=[-2])


def irft(u: torch.Tensor) -> torch.Tensor:
    """2D inverse real FFT along last two dimensions."""
    return irfftn(u, dim=[-2, -1])


# =============================================================================
# Kinetic Energy Spectrum
# =============================================================================

def compute_ke_spectrum_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D kinetic energy spectrum from velocity components.

    Args:
        u: U-velocity component
        v: V-velocity component

    Returns:
        Kinetic energy spectrum: (|u_fft|² + |v_fft|²) / 2
    """
    u_fft = rft(u)
    v_fft = rft(v)
    ke_spectrum = (u_fft * u_fft.conj() + v_fft * v_fft.conj()).real / 2.0
    return ke_spectrum


def spectral_loss_vec(
    u_pred: torch.Tensor,
    v_pred: torch.Tensor,
    u_true: torch.Tensor,
    v_true: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute spectral loss between predicted and true velocity fields.

    Note: This function has a bug (uses undefined variables). Use spectral_loss instead.
    """
    ke_pred = compute_ke_spectrum_2d(u_pred, v_pred)
    ke_true = compute_ke_spectrum_2d(u_true, v_true)
    log_ke_pred = torch.log(ke_pred + eps)
    log_ke_true = torch.log(ke_true + eps)
    return F.mse_loss(log_ke_pred, log_ke_true)


def create_boundary_mask(shape: tuple, boundary_width: int = 4) -> torch.Tensor:
    """
    Create a mask that zeros out boundary points.

    Args:
        shape: (height, width) of the mask
        boundary_width: Number of pixels to mask at each boundary

    Returns:
        Binary mask tensor with zeros at boundaries
    """
    mask = torch.ones(shape, dtype=torch.float32)
    mask[:boundary_width, :] = 0
    mask[-boundary_width:, :] = 0
    mask[:, :boundary_width] = 0
    mask[:, -boundary_width:] = 0
    return mask


# =============================================================================
# Main Spectral Loss Functions
# =============================================================================

def spectral_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    tukey: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute spectral loss in log-space of kinetic energy spectrum.

    Applies a Tukey window to reduce spectral leakage from boundaries,
    then computes MSE between log kinetic energy spectra.

    Args:
        y_pred: Predicted velocity field (B, 2, H, W) or (2, H, W)
        y_true: Target velocity field (B, 2, H, W) or (2, H, W)
        tukey: 2D Tukey window for tapering (H, W)
        eps: Small constant for numerical stability in log

    Returns:
        MSE loss in log-spectral space
    """
    if len(y_pred) == 3:
        u_pred, v_pred = y_pred[0, ...], y_pred[1, ...]
        u_true, v_true = y_true[0, ...], y_true[1, ...]
        u_pred = u_pred *  tukey
        v_pred = v_pred *  tukey
        u_true = u_true *  tukey
        v_true = v_true *  tukey
    else:
        u_pred, v_pred = y_pred[:,0, ...], y_pred[:,1, ...]
        u_true, v_true = y_true[:,0, ...], y_true[:,1, ...]
        u_pred = u_pred *  tukey[None, :, :]
        v_pred = v_pred *  tukey[None, :, :] 
        u_true = u_true *  tukey[None, :, :] 
        v_true = v_true *  tukey[None, :, :] 
    
    # Apply mask and window
    
    ke_pred = compute_ke_spectrum_2d(u_pred, v_pred)
    ke_true = compute_ke_spectrum_2d(u_true, v_true)
    log_ke_pred = torch.log(ke_pred + eps)
    log_ke_true = torch.log(ke_true + eps)
    return F.mse_loss(log_ke_pred, log_ke_true)

# =============================================================================
# Mirror-Padded Spectral Functions
# =============================================================================

def mirror_ke_spectrum_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D kinetic energy spectrum with mirror padding.

    Mirror padding reduces boundary artifacts by reflecting the field
    at boundaries before computing FFT.

    Args:
        u: U-velocity component (batch, height, width)
        v: V-velocity component (batch, height, width)

    Returns:
        Kinetic energy spectrum with reduced boundary effects
    """
    pad_h = u.shape[-2] - 1  # height padding
    pad_w = u.shape[-1] - 1  # width padding
    
    # Pad with (pad_left, pad_right, pad_top, pad_bottom)
    pad_sizes = (pad_w, pad_w, pad_h, pad_h)
    
    u_pad = torch.nn.functional.pad(u, pad_sizes, mode='reflect')
    v_pad = torch.nn.functional.pad(v, pad_sizes, mode='reflect') 
    # FFT of 3Nx3N field
    u_fft = rft(u_pad)
    v_fft = rft(v_pad)
    
    # Original spectrum would be (Nx//2 + 1, Ny//2 + 1)
    nx, ny = u.shape[-2], u.shape[-1]
    kx_max = nx//2 + 1
    ky_max = ny//2 + 1
    
    # Extract exactly the same number of modes as direct spectrum
    u_fft = u_fft[..., :kx_max, :ky_max]  # This gives us (Nx//2 + 1, Ny//2 + 1)
    v_fft = v_fft[..., :kx_max, :ky_max]
    
    # Compute energy spectrum
    ke_spectrum = (u_fft * u_fft.conj() + v_fft * v_fft.conj()).real / 2.0
    
    # Scale the spectrum to account for padding
    ke_spectrum *= 9
    
    return ke_spectrum

def spectral_loss_mirror(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
) -> torch.Tensor:
    """
    Compute spectral loss using mirror-padded FFT.

    Alternative to spectral_loss that uses mirror padding instead of
    Tukey windowing to handle boundary effects.

    Args:
        y_pred: Predicted velocity field (B, 2, H, W) or (2, H, W)
        y_true: Target velocity field (B, 2, H, W) or (2, H, W)

    Returns:
        MSE loss in log-spectral space
    """
    if len(y_pred) == 3:
        u_pred, v_pred = y_pred[0, ...], y_pred[1, ...]
        u_true, v_true = y_true[0, ...], y_true[1, ...]
    else:
        u_pred, v_pred = y_pred[:,0, ...], y_pred[:,1, ...]
        u_true, v_true = y_true[:,0, ...], y_true[:,1, ...]
    
    ke_pred = mirror_ke_spectrum_2d(u_pred, v_pred)
    ke_true = mirror_ke_spectrum_2d(u_true, v_true)
    log_ke_pred = torch.log(ke_pred)
    log_ke_true = torch.log(ke_true)
    return F.mse_loss(log_ke_pred, log_ke_true)




# =============================================================================
# Directional Spectral Functions
# =============================================================================

def mirror_ke_spectrum_x(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D kinetic energy spectrum in x-direction with mirror padding.

    Args:
        u: U-velocity component (batch, height, width)
        v: V-velocity component (batch, height, width)

    Returns:
        Kinetic energy spectrum along x-direction
    """
    pad_w = u.shape[-1] - 1  # width padding for x direction
    # Pad with (pad_left, pad_right)
    pad_sizes = (pad_w, pad_w, 0, 0)  # only pad in x direction
    u_pad = torch.nn.functional.pad(u, pad_sizes, mode='reflect')
    v_pad = torch.nn.functional.pad(v, pad_sizes, mode='reflect')
    
    # FFT only along x dimension
    u_fft = rft1(u_pad)  # This gives modes up to 3N/2
    v_fft = rft1(v_pad)
    
    # Original spectrum would be (N, N//2 + 1)
    nx = u.shape[-1]
    kx_max = nx//2 + 1
    
    # Extract exactly the same number of modes as direct spectrum
    u_fft = u_fft[..., :kx_max]  # Truncate to N/2 modes
    v_fft = v_fft[..., :kx_max]
    
    # Compute energy spectrum
    ke_spectrum = (u_fft * u_fft.conj() + v_fft * v_fft.conj()).real / 2.0
    
    # Scale the spectrum to account for padding (factor of 3 due to tripling in x)
    fac = (nx/2)**2
    ke_spectrum *= 3
    return ke_spectrum/fac

def mirror_ke_spectrum_y(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D kinetic energy spectrum in y-direction with mirror padding.

    Args:
        u: U-velocity component (batch, height, width)
        v: V-velocity component (batch, height, width)

    Returns:
        Kinetic energy spectrum along y-direction
    """
    pad_h = u.shape[-2] - 1  # height padding for y direction
    # Pad with (pad_top, pad_bottom)
    pad_sizes = (0, 0, pad_h, pad_h)  # only pad in y direction
    u_pad = torch.nn.functional.pad(u, pad_sizes, mode='reflect')
    v_pad = torch.nn.functional.pad(v, pad_sizes, mode='reflect')
    
    # FFT only along y dimension
    u_fft = rft2(u_pad)  # This gives modes up to 3N/2
    v_fft = rft2(v_pad)
    
    # Original spectrum would be (N//2 + 1, N)
    ny = u.shape[-2]
    ky_max = ny//2 + 1
    
    # Extract exactly the same number of modes as direct spectrum
    u_fft = u_fft[..., :ky_max, :]  # Truncate to N/2 modes
    v_fft = v_fft[..., :ky_max, :]
    
    # Compute energy spectrum
    ke_spectrum = (u_fft * u_fft.conj() + v_fft * v_fft.conj()).real / 2.0
    
    # Scale the spectrum to account for padding (factor of 3 due to tripling in y)
    ke_spectrum *= 3
    fac = (ny/2)**2
    return ke_spectrum/fac
def ident(x: torch.Tensor) -> torch.Tensor:
    """Identity function."""
    return x


def spectral_loss_directional(
    y_pred: torch.Tensor,
    y_true: torch.Tensor
) -> torch.Tensor:
    """
    Compute spectral loss separately in x and y directions.

    Useful for anisotropic flows where x and y spectral characteristics differ.

    Args:
        y_pred: Predicted velocity field (B, 2, H, W) or (2, H, W)
        y_true: Target velocity field (B, 2, H, W) or (2, H, W)

    Returns:
        Sum of x-direction and y-direction spectral losses
    """
    if len(y_pred.shape) == 3:
        u_pred, v_pred = y_pred[0, ...], y_pred[1, ...]
        u_true, v_true = y_true[0, ...], y_true[1, ...]
    else:
        u_pred, v_pred = y_pred[:,0, ...], y_pred[:,1, ...]
        u_true, v_true = y_true[:,0, ...], y_true[:,1, ...]
    func = ident 
    # Compute x-direction spectrum
    ke_x_pred = func(mirror_ke_spectrum_x(u_pred, v_pred))
    ke_x_true = func(mirror_ke_spectrum_x(u_true, v_true))
    
    # Compute y-direction spectrum
    ke_y_pred = func(mirror_ke_spectrum_y(u_pred, v_pred))
    ke_y_true = func(mirror_ke_spectrum_y(u_true, v_true))
    
    # Compute MSE for both spectra (no log needed)
    loss_x = F.mse_loss(ke_x_pred, ke_x_true)
    loss_y = F.mse_loss(ke_y_pred, ke_y_true)
    
    # Return sum of both losses
    return loss_x + loss_y

