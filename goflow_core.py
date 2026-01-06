"""
GOFLOW Core Utilities
=====================
Core functionality for GOFLOW (Geostationary Ocean Flow) model training:
- Gradient operators and derived field computation
- Loss functions (masked, gradient-based, spectral)
- Data loading utilities
- Model initialization and I/O
- Windowing and masking utilities

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from scipy.signal.windows import tukey
from sklearn.metrics import r2_score as R2

# Model imports - adjust paths as needed
from unet_vel_bn import UNet
from samudraUnet import SamudraUNet
from simpleCNN import TwoLayerCNN


# =============================================================================
# Gradient Operators
# =============================================================================

def dx_kernel(pm: float) -> torch.Tensor:
    """Create x-derivative kernel scaled by grid metric pm."""
    return pm * torch.Tensor([-1., 0., 1.]).view(1, 1, 1, 3) / 2


def dy_kernel(pn: float) -> torch.Tensor:
    """Create y-derivative kernel scaled by grid metric pn."""
    return pn * torch.Tensor([-1., 0., 1.]).view(1, 1, 3, 1) / 2


def compute_dx(field: torch.Tensor, kernel_x: torch.Tensor) -> torch.Tensor:
    """Compute x-derivative with replicate boundary padding."""
    return F.conv2d(F.pad(field, (1, 1, 0, 0), mode='replicate'), kernel_x)


def compute_dy(field: torch.Tensor, kernel_y: torch.Tensor) -> torch.Tensor:
    """Compute y-derivative with replicate boundary padding."""
    return F.conv2d(F.pad(field, (0, 0, 1, 1), mode='replicate'), kernel_y)


def compute_velocity_gradients(
    uv: torch.Tensor, 
    kernel_x: torch.Tensor, 
    kernel_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all four velocity gradient components.
    
    Args:
        uv: Velocity field tensor of shape (B, 2, H, W) with u in channel 0, v in channel 1
        kernel_x: X-derivative kernel
        kernel_y: Y-derivative kernel
        
    Returns:
        Tuple of (du/dx, du/dy, dv/dx, dv/dy)
    """
    u, v = uv[:, 0:1], uv[:, 1:2]
    ux = compute_dx(u, kernel_x)
    uy = compute_dy(u, kernel_y)
    vx = compute_dx(v, kernel_x)
    vy = compute_dy(v, kernel_y)
    return ux, uy, vx, vy


def compute_derived_fields(
    ux: torch.Tensor, 
    uy: torch.Tensor, 
    vx: torch.Tensor, 
    vy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute vorticity, divergence, and strain magnitude from velocity gradients.
    
    Returns:
        Tuple of (vorticity, divergence, strain_magnitude)
    """
    vorticity = vx - uy
    divergence = ux + vy
    s1 = ux - vy  # Normal strain
    s2 = vx + uy  # Shear strain
    strain = torch.sqrt(s1**2 + s2**2)
    return vorticity, divergence, strain


# =============================================================================
# Window and Mask Creation
# =============================================================================

def create_tukey_window(shape: tuple, alpha: float = 0.5) -> torch.Tensor:
    """
    Create 2D Tukey (tapered cosine) window for spectral analysis.
    
    Args:
        shape: (height, width) of the window
        alpha: Shape parameter (0 = rectangular, 1 = Hann window)
    """
    shape = tuple(shape)
    window_y = torch.from_numpy(tukey(shape[0], alpha)).float()
    window_x = torch.from_numpy(tukey(shape[1], alpha)).float()
    return torch.outer(window_y, window_x)


def create_boundary_mask(shape: tuple, boundary_width: int = 2) -> torch.Tensor:
    """
    Create a mask that zeros out boundary points.
    
    Args:
        shape: (height, width) of the mask
        boundary_width: Number of pixels to mask at each boundary
    """
    mask = torch.ones(tuple(shape), dtype=torch.float32)
    mask[:boundary_width, :] = 0
    mask[-boundary_width:, :] = 0
    mask[:, :boundary_width] = 0
    mask[:, -boundary_width:] = 0
    return mask


# =============================================================================
# Loss Functions
# =============================================================================

def masked_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor, 
    criterion: nn.Module
) -> torch.Tensor:
    """Compute loss with boundary masking, normalized by mask coverage."""
    return criterion(pred * mask, target * mask) / mask.mean()


def gradient_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    criterion: nn.Module, 
    mask: torch.Tensor, 
    kernel_x: torch.Tensor, 
    kernel_y: torch.Tensor,
    weights: tuple = (1.0, 1.0, 1.0)
) -> torch.Tensor:
    """
    Compute loss on derived gradient fields (vorticity, divergence, strain).
    
    Args:
        pred: Predicted velocity field (B, 2, H, W)
        target: Target velocity field (B, 2, H, W)
        criterion: Loss function (e.g., L1Loss)
        mask: Boundary mask
        kernel_x, kernel_y: Derivative kernels
        weights: Tuple of (vort_weight, div_weight, strain_weight)
    """
    # Compute gradients
    ux1, uy1, vx1, vy1 = compute_velocity_gradients(pred, kernel_x, kernel_y)
    ux2, uy2, vx2, vy2 = compute_velocity_gradients(target, kernel_x, kernel_y)
    
    # Compute derived fields
    vort1, div1, strain1 = compute_derived_fields(ux1, uy1, vx1, vy1)
    vort2, div2, strain2 = compute_derived_fields(ux2, uy2, vx2, vy2)
    
    # Apply mask (add batch/channel dimensions)
    mask_bc = mask[None, :, :]
    
    loss = (weights[0] * criterion(vort1 * mask_bc, vort2 * mask_bc) +
            weights[1] * criterion(div1 * mask_bc, div2 * mask_bc) +
            weights[2] * criterion(strain1 * mask_bc, strain2 * mask_bc))
    
    return loss


# =============================================================================
# Data Loading
# =============================================================================

def load_datasets(
    llc_file: str,
    varlist: list,
    train_inds: list,
    test_inds: tuple,
    valid_inds: tuple = None,
    step0: int = 1,
    nframes: int = 3
):
    """
    Load and prepare train/test/validation datasets.
    
    Args:
        llc_file: Path to LLC NetCDF file
        varlist: List of variable names to load
        train_inds: List of index tuples for training regions
        test_inds: Index tuple for test region
        valid_inds: Optional index tuple for validation region
        step0: Time step stride
        nframes: Number of input frames
        
    Returns:
        Tuple of (train_dataset, test_dataset) or (train, test, valid) if valid_inds provided
    """
    from dataSST import SSTDataset
    
    trainlist = [
        SSTDataset(llc_file, varlist, inds, step0=step0, num_input_frames=nframes)
        for inds in train_inds
    ]
    train = ConcatDataset(trainlist)
    test = SSTDataset(llc_file, varlist, test_inds, step0=step0, num_input_frames=nframes)
    
    if valid_inds is not None:
        valid = SSTDataset(llc_file, varlist, valid_inds, step0=step0, num_input_frames=nframes)
        return train, test, valid
    return train, test


def create_dataloaders(
    train_dataset,
    test_dataset,
    valid_dataset=None,
    batch_sizes: dict = None,
    num_workers: int = 5
):
    """
    Create DataLoader objects for train/test/validation.
    
    Args:
        batch_sizes: Dict with keys 'train', 'test', and optionally 'valid'
    """
    if batch_sizes is None:
        batch_sizes = {'train': 64, 'test': 200, 'valid': 50}
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_sizes['train'], 
        shuffle=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_sizes['test'], 
        shuffle=False, 
        num_workers=num_workers
    )
    
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_sizes['valid'], 
            shuffle=False, 
            num_workers=num_workers
        )
        return train_loader, test_loader, valid_loader
    
    return train_loader, test_loader


# =============================================================================
# Model Initialization and I/O
# =============================================================================

# Model name mappings
MODEL_CONFIGS = {
    'unet': {'class': UNet, 'default_nbase': 16},
    'samudra0': {'class': SamudraUNet, 'padding_mode': 'zeros'},
    'samudraR': {'class': SamudraUNet, 'padding_mode': 'reflect'},
    '2layer': {'class': TwoLayerCNN, 'default_kernel_size': 5},
}


def get_model_string(model_name: str, nbase: int = 16, kernel_size: int = 5, use_grad: bool = False) -> str:
    """Generate model filename string based on configuration."""
    prefix = 'lgt_'
    if model_name == 'unet':
        return f'{prefix}unet{nbase}'
    elif model_name in ('samudra0', 'samudraR'):
        suffix = '0' if model_name == 'samudra0' else 'R'
        return f'{prefix}sam{suffix}'
    elif model_name == '2layer':
        return f'{prefix}{kernel_size}x{kernel_size}'
    return f'{prefix}{model_name}'


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(
    n_input: int,
    n_output: int,
    model_name: str = 'unet',
    nbase: int = 16,
    kernel_size: int = 5,
    device: torch.device = None,
    inp_norm: bool = True
) -> nn.Module:
    """
    Initialize a GOFLOW model.

    Args:
        n_input: Number of input channels
        n_output: Number of output channels
        model_name: One of 'unet', 'samudra0', 'samudraR', '2layer'
        nbase: Base channel count for UNet
        kernel_size: Kernel size for 2layer CNN
        device: Target device
        inp_norm: Whether to apply input batch normalization (default True for training,
                  set False for inference with legacy models trained without normalization)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'unet':
        model = UNet(n_input, n_output, bilinear=True, Nbase=nbase, inpNorm=inp_norm)
    elif model_name == 'samudra0':
        model = SamudraUNet(n_channels=3, no=2, Nbase=2, padding_mode='zeros', inpNorm=inp_norm)
    elif model_name == 'samudraR':
        model = SamudraUNet(n_channels=3, no=2, Nbase=2, padding_mode='reflect', inpNorm=inp_norm)
    elif model_name == '2layer':
        model = TwoLayerCNN(kernel_size=kernel_size, inpNorm=inp_norm)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    model = model.to(device)
    n_params = count_parameters(model)
    print(f'Model {model_name} initialized with {n_params/1e6:.2f}M parameters')
    
    return model


def save_model(model: nn.Module, filepath: str):
    """Save model state dict."""
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


def load_model(model: nn.Module, filepath: str, device: torch.device = None) -> nn.Module:
    """Load model state dict with device handling."""
    print(f'Loading model from {filepath}...')
    if device is None:
        device = next(model.parameters()).device
    
    try:
        model.load_state_dict(torch.load(filepath, map_location=device))
    except RuntimeError as e:
        print(f'Warning: {e}')
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
    
    return model


# =============================================================================
# Evaluation Utilities
# =============================================================================

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to flattened numpy array."""
    return x.cpu().numpy().flatten()


def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Compute R² score, optionally with masking."""
    if mask is not None:
        y_true = y_true * mask
        y_pred = y_pred * mask
    return R2(to_numpy(y_true), to_numpy(y_pred))


def compute_gradient_r2(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    mask: torch.Tensor
) -> float:
    """
    Compute R² score on vorticity and strain fields.
    
    Returns average of vorticity R² and strain R².
    """
    # Compute gradients
    ux_true, uy_true, vx_true, vy_true = compute_velocity_gradients(y_true, kernel_x, kernel_y)
    ux_pred, uy_pred, vx_pred, vy_pred = compute_velocity_gradients(y_pred, kernel_x, kernel_y)
    
    # Compute derived fields
    vort_true, _, strain_true = compute_derived_fields(ux_true, uy_true, vx_true, vy_true)
    vort_pred, _, strain_pred = compute_derived_fields(ux_pred, uy_pred, vx_pred, vy_pred)
    
    # Compute R² for vorticity and strain
    r2_vort = R2(to_numpy(vort_true * mask), to_numpy(vort_pred * mask))
    r2_strain = R2(to_numpy(strain_true * mask), to_numpy(strain_pred * mask))
    
    return (r2_vort + r2_strain) / 2.0
