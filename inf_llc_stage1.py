"""
GOFLOW Inference Script
=======================
Inference pipeline for running trained GOFLOW models on test and satellite data.

Usage:
    python inf_llc_stage1.py --model_file lgt_unet16_1_3_0.2cs.pth --goes_file GS_data.nc

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from netCDF4 import Dataset as NCDataset
from tqdm import tqdm

# Local imports
from goflow_core import (
    dx_kernel, dy_kernel,
    compute_velocity_gradients, compute_derived_fields,
    create_boundary_mask,
    load_datasets, create_dataloaders,
    initialize_model, load_model, count_parameters
)
from dataSST import SatelliteDataset, writeGridSat
from writenc import ncCreate, addVal


# =============================================================================
# Configuration
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run GOFLOW inference')

    # Device configuration
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')

    # Model configuration
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--nbase', type=int, default=16,
                        help='Base channels for UNet (must match training)')
    parser.add_argument('--inp_norm', action='store_true',
                        help='Use input normalization (set if model was trained with it)')

    # Data paths
    parser.add_argument('--goes_files', type=str, nargs='+',
                        default=['GS_BT_NESMA2023_HiRes_SUBSECTION_grad_mask.nc'],
                        help='GOES satellite data files to process')
    parser.add_argument('--llc_file', type=str, default='llcGoes_gradT_trunc.nc',
                        help='LLC data file for test set evaluation')
    parser.add_argument('--output_dir', type=str, default='./ncfiles/',
                        help='Output directory for NetCDF files')

    # Processing parameters
    parser.add_argument('--pm', type=float, default=5.0, help='X grid metric')
    parser.add_argument('--pn', type=float, default=5.0, help='Y grid metric')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for satellite inference')

    # Test result blending (for analysis)
    parser.add_argument('--blend_alpha', type=float, default=0.5,
                        help='Blend factor for test results: alpha*true + (1-alpha)*pred')

    # Flags
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip test set evaluation')
    parser.add_argument('--skip_satellite', action='store_true',
                        help='Skip satellite data processing')

    return parser.parse_args()


def setup_device(cuda_idx: int) -> torch.device:
    """Configure CUDA device."""
    device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_idx)
    print(f'Device: {device}')
    return device


# =============================================================================
# Evaluation Functions
# =============================================================================

def test_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor
) -> float:
    """
    Compute R² score for a single batch.

    Args:
        model: Trained model
        x: Input tensor
        y: Target tensor

    Returns:
        R² score
    """
    from sklearn.metrics import r2_score as R2

    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        r2 = R2(y.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())
    return r2


def test_step_batch(
    model: torch.nn.Module,
    test_loader: DataLoader,
    mask: torch.Tensor
) -> float:
    """
    Compute average R² score over entire test set with masking.

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        mask: Boundary mask tensor

    Returns:
        Average R² score
    """
    from sklearn.metrics import r2_score as R2

    model.eval()
    device = next(model.parameters()).device
    total_r2 = 0.0
    count = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Testing'):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            y_masked = (y * mask[None, None, :, :]).cpu().numpy().flatten()
            y_pred_masked = (y_pred * mask[None, None, :, :]).cpu().numpy().flatten()
            r2 = R2(y_masked, y_pred_masked)

            total_r2 += r2
            count += 1

    return total_r2 / count


# =============================================================================
# Satellite Data Processing
# =============================================================================

def process_satellite_data(
    model: torch.nn.Module,
    goes_file: str,
    valid_inds: tuple,
    pm: float,
    pn: float,
    batch_size: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on GOES satellite data.

    Args:
        model: Trained model
        goes_file: Path to GOES NetCDF file
        valid_inds: Spatial indices (y0, y1, x0, x1)
        pm: X grid metric for gradient computation
        pn: Y grid metric for gradient computation
        batch_size: Batch size for inference

    Returns:
        Tuple of (velocities, gradient_fields, sst_data) as numpy arrays
    """
    device = next(model.parameters()).device
    kernel_x = dx_kernel(pm).to(device)
    kernel_y = dy_kernel(pn).to(device)

    goes_dataset = SatelliteDataset(goes_file, ['log_gradT'], valid_inds, train=False)
    goes_loader = DataLoader(goes_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    out_list = []
    grad_list = []
    sst_list = []

    model.eval()
    with torch.no_grad():
        for sst in tqdm(goes_loader, desc='Satellite inference'):
            # Store middle frame of input SST
            sst_list.append(sst[:, 1, :, :].cpu().numpy()[:, None, :, :])

            sst = sst.to(device)
            out = model(sst)
            out_list.append(out.cpu().numpy())

            # Compute gradient fields
            ux, uy, vx, vy = compute_velocity_gradients(out, kernel_x, kernel_y)
            vort, div, strain = compute_derived_fields(ux, uy, vx, vy)
            grad_list.append(torch.stack((vort, div, strain), dim=1).cpu().numpy())

    out_val = np.concatenate(out_list, axis=0)
    grad_val = np.concatenate(grad_list, axis=0)
    sst_val = np.concatenate(sst_list, axis=0).squeeze()

    return out_val, grad_val, sst_val


# =============================================================================
# NetCDF Output
# =============================================================================

def write_satellite_netcdf(
    output_file: str,
    out_val: np.ndarray,
    grad_val: np.ndarray,
    sst_val: np.ndarray,
    valid_inds: tuple,
    goes_file: str
):
    """
    Write satellite prediction results to NetCDF.

    Args:
        output_file: Output NetCDF filename
        out_val: Velocity predictions (nt, 2, ny, nx)
        grad_val: Gradient fields (nt, 3, ny, nx) - vort, div, strain
        sst_val: Input SST data (nt, ny, nx)
        valid_inds: Spatial indices used for subsetting
        goes_file: Source GOES file for metadata
    """
    nt, _, ny, nx = out_val.shape
    print(f'Writing {output_file}: shape=({nt}, {ny}, {nx})')

    with NCDataset(goes_file, 'r') as nch:
        varnames = ['U', 'V', 'Vorticity', 'Divergence', 'Strain', 'BT', 'loggrad_BT']
        nc = ncCreate(output_file, nx, ny, varnames, dt=2)

        for it in tqdm(range(nt), desc='Writing NetCDF'):
            BT = nch.variables['BT'][it + 12,
                                     valid_inds[0]:valid_inds[1],
                                     valid_inds[2]:valid_inds[3]]
            mask = nch.variables['mask'][it + 12,
                                         valid_inds[0]:valid_inds[1],
                                         valid_inds[2]:valid_inds[3]]

            addVal(nc, 'U', out_val[it, 0, :, :] * mask, it)
            addVal(nc, 'V', out_val[it, 1, :, :] * mask, it)
            addVal(nc, 'Vorticity', grad_val[it, 0, :, :] * mask, it)
            addVal(nc, 'Divergence', grad_val[it, 1, :, :] * mask, it)
            addVal(nc, 'Strain', grad_val[it, 2, :, :] * mask, it)
            addVal(nc, 'BT', BT * mask, it)
            addVal(nc, 'loggrad_BT', sst_val[it, :, :] * mask, it)

        nc.close()

    writeGridSat(goes_file, output_file, valid_inds)


def write_test_results(
    model: torch.nn.Module,
    test_loader: DataLoader,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    output_dir: str = './ncfiles/',
    blend_alpha: float = 0.5
):
    """
    Write test set predictions and gradient fields to NetCDF.

    The output includes both true and predicted fields, with predictions
    optionally blended with ground truth for analysis purposes.

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        kernel_x: X-derivative kernel
        kernel_y: Y-derivative kernel
        output_dir: Output directory
        blend_alpha: Blend factor (alpha*true + (1-alpha)*pred). Set to 0 for pure predictions.
    """
    model.eval()
    device = next(model.parameters()).device

    inputs_list = []
    outputs_list = []
    targets_list = []
    true_grads_list = []
    pred_grads_list = []

    with torch.no_grad():
        for x, y_true in tqdm(test_loader, desc='Processing test set'):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)

            # Blend prediction with ground truth for analysis
            y_blended = blend_alpha * y_true + (1 - blend_alpha) * y_pred

            # Compute gradients for true and blended fields
            ux_true, uy_true, vx_true, vy_true = compute_velocity_gradients(
                y_true, kernel_x, kernel_y)
            ux_pred, uy_pred, vx_pred, vy_pred = compute_velocity_gradients(
                y_blended, kernel_x, kernel_y)

            vort_true, div_true, strain_true = compute_derived_fields(
                ux_true, uy_true, vx_true, vy_true)
            vort_pred, div_pred, strain_pred = compute_derived_fields(
                ux_pred, uy_pred, vx_pred, vy_pred)

            inputs_list.append(x[:, 1, :, :].cpu().numpy())
            outputs_list.append(y_blended.cpu().numpy())
            targets_list.append(y_true.cpu().numpy())
            true_grads_list.append(
                torch.stack((vort_true, div_true, strain_true), dim=1).cpu().numpy())
            pred_grads_list.append(
                torch.stack((vort_pred, div_pred, strain_pred), dim=1).cpu().numpy())

    # Concatenate batches
    inputs = np.concatenate(inputs_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    true_grads = np.concatenate(true_grads_list, axis=0)
    pred_grads = np.concatenate(pred_grads_list, axis=0)

    # Write NetCDF
    os.makedirs(output_dir, exist_ok=True)
    nc_filename = os.path.join(output_dir, 'test_results_epoch.nc')

    Nt, Ny, Nx = inputs.shape
    varlist = ['gradT', 'U_inp', 'V_inp', 'vort_inp', 'div_inp', 'strain_inp',
               'U_out', 'V_out', 'vort_out', 'div_out', 'strain_out']

    with ncCreate(nc_filename, Nx, Ny, varlist) as nc:
        nc.variables['gradT'][:] = inputs
        nc.variables['U_inp'][:] = targets[:, 0, :, :]
        nc.variables['V_inp'][:] = targets[:, 1, :, :]
        nc.variables['U_out'][:] = outputs[:, 0, :, :]
        nc.variables['V_out'][:] = outputs[:, 1, :, :]
        nc.variables['vort_inp'][:] = true_grads[:, 0, :, :]
        nc.variables['div_inp'][:] = true_grads[:, 1, :, :]
        nc.variables['strain_inp'][:] = true_grads[:, 2, :, :]
        nc.variables['vort_out'][:] = pred_grads[:, 0, :, :]
        nc.variables['div_out'][:] = pred_grads[:, 1, :, :]
        nc.variables['strain_out'][:] = pred_grads[:, 2, :, :]

        nc.description = f'Test set results (blend_alpha={blend_alpha})'
        nc.input_field = 'SST gradient (middle time step)'
        nc.output_fields = 'Vorticity, Divergence, Strain (input and output)'

    print(f'Test results written to {nc_filename}')


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    args = parse_args()

    # Setup device
    device = setup_device(args.cuda)

    # Get grid dimensions from first GOES file
    with NCDataset(args.goes_files[0], 'r') as nc:
        Nx = nc.dimensions['lon'].size
        Ny = nc.dimensions['lat'].size

    # Define spatial indices
    valid_inds = (0, 512, Nx - 768, Nx)
    test_inds = (0, 256, Nx - 256, Nx)

    # Initialize model (inp_norm=False for legacy models by default)
    ni, no = 3, 2
    model = initialize_model(
        ni, no,
        model_name='unet',
        nbase=args.nbase,
        device=device,
        inp_norm=args.inp_norm
    )

    # Load trained weights
    model = load_model(model, args.model_file, device)
    print(f'Loaded model from {args.model_file}')

    # Setup derivative kernels
    kernel_x = dx_kernel(args.pm).to(device)
    kernel_y = dy_kernel(args.pn).to(device)

    # Process test set if requested
    if not args.skip_test and os.path.exists(args.llc_file):
        print('\n--- Processing Test Set ---')
        varlist = ['loggrad_T', 'U', 'V']
        train_inds = [(0, 256, 256, 512)]  # Minimal for loading

        _, test_data = load_datasets(
            args.llc_file, varlist, train_inds, test_inds,
            step0=1, nframes=3
        )
        _, test_loader = create_dataloaders(
            test_data, test_data,
            batch_sizes={'train': 64, 'test': 200, 'valid': 50}
        )

        # Compute R² on test set
        sample_y = next(iter(test_loader))[1]
        mask = create_boundary_mask(sample_y.shape[-2:]).to(device)
        r2 = test_step_batch(model, test_loader, mask)
        print(f'Test R²: {r2:.4f}')

        # Write test results
        write_test_results(
            model, test_loader, kernel_x, kernel_y,
            output_dir=args.output_dir,
            blend_alpha=args.blend_alpha
        )

    # Process satellite data
    if not args.skip_satellite:
        print('\n--- Processing Satellite Data ---')
        for goes_file in args.goes_files:
            if not os.path.exists(goes_file):
                print(f'Warning: {goes_file} not found, skipping')
                continue

            print(f'Processing {goes_file}...')
            output_file = f'preds_{os.path.basename(args.model_file).replace(".pth", "")}_{goes_file}'

            out_val, grad_val, sst_val = process_satellite_data(
                model, goes_file, valid_inds,
                pm=args.pm, pn=args.pn,
                batch_size=args.batch_size
            )

            write_satellite_netcdf(
                output_file, out_val, grad_val, sst_val,
                valid_inds, goes_file
            )

    print('\nInference complete!')


if __name__ == '__main__':
    main()
