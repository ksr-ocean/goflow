"""
SST Dataset Classes
====================
PyTorch Dataset implementations for loading SST and satellite data for GOFLOW training.

Includes:
- SSTDataset: LLC model data (SST gradients + velocity fields)
- SatelliteDataset: GOES satellite brightness temperature data
- SSTDatasetTime: Time-range variant of SSTDataset

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import numpy as np
from netCDF4 import Dataset as load
from torch.utils.data import DataLoader, Dataset, ConcatDataset


# =============================================================================
# Normalization Constants
# =============================================================================

# Log SST gradient normalization range
lgtMax = 0
lgtMin = -19

# SST normalization range (Celsius)
sstMax = 30
sstMin = 0


# =============================================================================
# Grid Parameter Loading
# =============================================================================

def loadGridParams(nc_file: str) -> tuple:
    """
    Load grid parameters from NetCDF file for use as additional model inputs.

    Computes various combinations of lat/lon encoded as sin/cos,
    plus log bathymetry for geographic conditioning.

    Args:
        nc_file: Path to NetCDF grid file

    Returns:
        Tuple of (Phi, PhiLat, PhiLgh, PhiLatLgh, lgh) arrays
    """
    nch = load(nc_file, 'r')
    try:
        lon, lat = nch.variables['lon'], nch.variables['lat']
        lon2d, lat2d = np.meshgrid(lon, lat)
    except:
        lon = nch.variables['lon_rho'][:].astype(np.float32)
        lat = nch.variables['lat_rho'][:].astype(np.float32)
        lon2d, lat2d = lon, lat

    h = nch.variables['h'][:].astype(np.float32)
    lgh = np.log10(1 + h) / 6

    Sphi = np.sin(lat2d * np.pi / 180)
    Cphi = np.cos(lat2d * np.pi / 180)
    Stheta = np.abs(np.sin(lon2d * np.pi / 180))
    Ctheta = np.abs(np.cos(lon2d * np.pi / 180))

    PhiLat = np.concatenate([Sphi[None, :, :], Cphi[None, :, :]], axis=0).astype(np.float32)
    Phi = np.concatenate([Sphi[None, :, :], Cphi[None, :, :],
                          Stheta[None, :, :], Ctheta[None, :, :]], axis=0).astype(np.float32)
    PhiLgh = np.concatenate([Phi, lgh[None, :, :]], axis=0).astype(np.float32)
    PhiLatLgh = np.concatenate([PhiLat, lgh[None, :, :]], axis=0).astype(np.float32)

    return Phi, PhiLat, PhiLgh, PhiLatLgh, lgh


# =============================================================================
# Data Access Helpers
# =============================================================================

def getGrid(gridStuff: np.ndarray, ij: tuple) -> np.ndarray:
    """Extract spatial subset from grid data."""
    return gridStuff[:, ij[0]:ij[1], ij[2]:ij[3]]


def getData(nch, var: str, ij: tuple, time_slice: int) -> np.ndarray:
    """Extract spatiotemporal subset from NetCDF variable."""
    return nch.variables[var][time_slice, ij[0]:ij[1], ij[2]:ij[3]]


# =============================================================================
# Dataset Classes
# =============================================================================

class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for GOES satellite brightness temperature data.

    Loads 3-frame sequences of satellite observations with optional velocity targets.
    Uses 12-hour spacing between frames (indices i, i+12, i+24).

    Args:
        nc_path: Path to satellite NetCDF file
        var_names: Variable names to load (e.g., ['log_gradT'] or ['BT'])
        spatial_slice: Tuple (y0, y1, x0, x1) for spatial subsetting
        train: If True, return (input, target) tuples; if False, return input only
        gridField: Optional grid parameters to concatenate with input
    """

    def __init__(self, nc_path, var_names, spatial_slice, train=True, gridField=None):
        self.nc_path = nc_path
        self.var_names = var_names
        self.train = train
        self.spatial_slice = spatial_slice
        self.gridField = gridField

        self.nch = load(self.nc_path, 'r')
        self.time_len = self.nch.dimensions['time'].size - 25

    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        # Adjust index to start from the second sample (idx + 1)
        idx = idx + 1

        # Get SST slices at i, i+12, and i+24
        sst_slices = [
            getData(self.nch, self.var_names[0], self.spatial_slice, idx),
            getData(self.nch, self.var_names[0], self.spatial_slice, idx + 12),
            getData(self.nch, self.var_names[0], self.spatial_slice, idx + 24)
        ]
        
        sst_slices = np.stack(sst_slices, axis=0)
        if self.var_names[0] == 'log_gradT':
            sst_slices = (sst_slices - lgtMin) / (lgtMax - lgtMin)
        else:
            sst_slices = (sst_slices - sstMin) / (sstMax - sstMin)

        #  
        #sst_slices[sst_slices == 1.0] = 0
        
        if self.gridField is not None:
            input_slice = np.concatenate([sst_slices, getGrid(self.gridField, self.spatial_slice)], axis=0).astype(np.float32)
        else:
            input_slice = np.nan_to_num(sst_slices.astype(np.float32))

        if self.train:
            # Get velocity fields at the middle time step (i+12)
            u_slice = getData(self.nch, self.var_names[1], self.spatial_slice, idx + 12)
            v_slice = getData(self.nch, self.var_names[2], self.spatial_slice, idx + 12)
            uv_slice = np.stack((u_slice, v_slice), axis=0).astype(np.float32)
            return input_slice, uv_slice
        else:
            return input_slice

class SSTDataset(Dataset):
    """
    PyTorch Dataset for LLC ocean model SST gradient and velocity data.

    Loads multi-frame SST gradient sequences with corresponding velocity targets.
    By default uses non-causal setup where target is at middle time step.

    Args:
        nc_path: Path to LLC NetCDF file (or list of two files for multi-source)
        var_names: Variable names ['loggrad_T', 'U', 'V'] or with optional SSH
        spatial_slice: Tuple (y0, y1, x0, x1) for spatial subsetting
        step0: Time step stride between frames
        num_input_frames: Number of input SST frames (typically 3)
        train: If True, return (input, target) tuples; if False, return input only
        gridField: Optional grid parameters to concatenate with input
        overlap: If True, use overlapping samples; if False, non-overlapping
    """

    def __init__(self, nc_path, var_names, spatial_slice, step0, num_input_frames,
                 train=True, gridField=None, overlap=False):
        self.nc_path = nc_path
        self.var_names = var_names
        self.train = train
        self.spatial_slice = spatial_slice
        self.step0 = step0
        self.num_input_frames = num_input_frames
        self.causal = False
        self.overlap = overlap
        self.sshFlag = len(var_names) > 3
        if isinstance(nc_path,list)==True:
            self.nch = load(self.nc_path[0], 'r')
            self.nch1 = load(self.nc_path[1], 'r')
            self.list = True
        else:
            self.nch = load(self.nc_path, 'r')
            self.list = False
        if self.overlap:
            self.time_len = (self.nch.dimensions['time'].size - num_input_frames*step0)
        else:
            self.time_len = (self.nch.dimensions['time'].size - num_input_frames*step0) // (num_input_frames * step0)

        self.gridField = gridField
    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        if self.overlap is False:
            idx = idx * self.num_input_frames * self.step0  # Adjust index for non-overlapping segments
        sst_slices = [getData(self.nch, self.var_names[0], self.spatial_slice, idx + i * self.step0) for i in range(self.num_input_frames)]
        #grid_slice =
        #normalize each input variable; possibly a lot more general
        
        if self.var_names[0] == 'loggrad_T':
            #sst_slices = (sst_slices - lgtMin) / (lgtMax - lgtMin)
            sst_slices = (np.stack(sst_slices, axis=0) - lgtMin)/(lgtMax - lgtMin)
        else:
            sst_slices = (np.stack(sst_slices, axis=0) - sstMin)/(sstMax - sstMin)
        
        #sst_slices[sst_slices==1.0] = 0
        
        if self.gridField is not None:
            input_slice = np.concatenate([sst_slices, getGrid(self.gridField, self.spatial_slice)], axis = 0).astype(np.float32)
        else:
            input_slice = np.nan_to_num(sst_slices.data.astype(np.float32))
        if self.train:
            if self.causal:
                u_slice = getData(self.nch, self.var_names[1], self.spatial_slice, idx + (self.num_input_frames -1) * self.step0)
                v_slice = getData(self.nch, self.var_names[2], self.spatial_slice, idx + (self.num_input_frames -1) * self.step0)
                if self.sshFlag:
                    ssh_slice = getData(self.nch, self.var_names[3], self.spatial_slice, idx + (self.num_input_frames -1) * self.step0)

            else:
                u_slice = getData(self.nch, self.var_names[1], self.spatial_slice, idx + (self.num_input_frames // 2) * self.step0)
                v_slice = getData(self.nch, self.var_names[2], self.spatial_slice, idx + (self.num_input_frames // 2) * self.step0)
                if self.sshFlag:
                    ssh_slice = getData(self.nch, self.var_names[3], self.spatial_slice, idx + (self.num_input_frames // 2) * self.step0)
            if self.sshFlag:
                uv_slice = np.stack((u_slice, v_slice, ssh_slice), axis=0).astype(np.float32)
            else:
                uv_slice = np.stack((u_slice, v_slice), axis=0).astype(np.float32)

            return input_slice, uv_slice
        else:
            return input_slice

class SSTDatasetTime(Dataset):
    """
    Time-range variant of SSTDataset with explicit time bounds.

    Similar to SSTDataset but allows specifying a time range within the file.
    Uses causal setup where target is at the last input frame time.

    Args:
        nc_path: Path to LLC NetCDF file
        var_names: Variable names ['loggrad_T', 'U', 'V']
        spatial_slice: Tuple (y0, y1, x0, x1) for spatial subsetting
        time_range: Tuple (start_time, end_time) for temporal subsetting
        step0: Time step stride between frames
        num_input_frames: Number of input SST frames
        train: If True, return (input, target) tuples; if False, return input only
        gridField: Optional grid parameters to concatenate with input
        overlap: Unused, kept for API compatibility
    """

    def __init__(self, nc_path, var_names, spatial_slice, time_range, step0,
                 num_input_frames, train=True, gridField=None, overlap=False):
        self.nc_path = nc_path
        self.var_names = var_names
        self.train = train
        self.spatial_slice = spatial_slice
        self.time_range = time_range
        self.step0 = step0
        self.num_input_frames = num_input_frames
        self.causal = True
        if isinstance(nc_path, list):
            self.nch = load(self.nc_path[0], 'r')
            self.nch1 = load(self.nc_path[1], 'r')
            self.list = True
        else:
            self.nch = load(self.nc_path, 'r')
            self.list = False
        self.time_len = (self.time_range[1] - self.time_range[0] - num_input_frames * step0) // (num_input_frames * step0)
        self.gridField = gridField

    def __len__(self):
        return self.time_len
    def __getitem__(self, idx):
        idx = self.time_range[0] + idx * self.num_input_frames * self.step0  # Adjust index for non-overlapping segments
        sst_slices = [getData(self.nch, self.var_names[0], self.spatial_slice, idx + i * self.step0) for i in range(self.num_input_frames)]
        sst_slices = (np.stack(sst_slices, axis=0) - lgtMin) / (lgtMax - lgtMin)
        sst_slices[sst_slices==1.0] = 0
        if self.gridField is not None:
            input_slice = np.concatenate([sst_slices, getGrid(self.gridField, self.spatial_slice)], axis=0).astype(np.float32)
        else:
            input_slice = np.nan_to_num(sst_slices.data.astype(np.float32))
        if self.train:
            if self.causal:
                u_slice = getData(self.nch, self.var_names[1], self.spatial_slice, idx + (self.num_input_frames - 1) * self.step0)
                v_slice = getData(self.nch, self.var_names[2], self.spatial_slice, idx + (self.num_input_frames - 1) * self.step0)
            uv_slice = np.stack((u_slice, v_slice), axis=0).astype(np.float32)
            return input_slice, uv_slice
        else:
            return input_slice
# =============================================================================
# NetCDF Grid Writing Utilities
# =============================================================================

from netCDF4 import Dataset


def writeGridSat(nc_file_path: str, target: str, spatial_slice: tuple):
    """
    Add coordinate variables to a satellite prediction NetCDF file.

    Copies time, lat, lon from source file to target file, adjusting
    time indices for the 3-frame sampling scheme.

    Args:
        nc_file_path: Source satellite data file
        target: Target prediction file to add coordinates to
        spatial_slice: Tuple (y0, y1, x0, x1) used for subsetting
    """
    nc = Dataset(nc_file_path, 'r')
    nco = Dataset(target, 'a')

    time = nc.variables['time'][:]
    lat = nc.variables['lat'][spatial_slice[0]:spatial_slice[1]]
    lon = nc.variables['lon'][spatial_slice[2]:spatial_slice[3]]
    print(len(lat), len(lon))

    # Time indices for middle frames in 3-frame sequences
    middle_steps = np.arange(1, len(time) - 2, dtype=int)
    middle_times = time[middle_steps]

    nco.createVariable('time', np.dtype('float32').char, ('time'))
    nco.createVariable('lon', np.dtype('float32').char, ('lon'))
    nco.createVariable('lat', np.dtype('float32').char, ('lat'))

    nco.variables['time'][:] = middle_times
    nco.variables['lon'][:] = lon
    nco.variables['lat'][:] = lat

    nco.close()
    nc.close()


def writeGridSatNoFrame(nc_file_path: str, target: str, spatial_slice: tuple):
    """
    Add coordinate variables without frame adjustment.

    Similar to writeGridSat but copies all time steps without the
    middle-frame adjustment used for 3-frame sequences.

    Args:
        nc_file_path: Source satellite data file
        target: Target prediction file to add coordinates to
        spatial_slice: Tuple (y0, y1, x0, x1) used for subsetting
    """
    nc = Dataset(nc_file_path, 'r')
    nco = Dataset(target, 'a')

    time = nc.variables['time'][:]
    lat = nc.variables['lat'][spatial_slice[0]:spatial_slice[1]]
    lon = nc.variables['lon'][spatial_slice[2]:spatial_slice[3]]
    print(len(lat), len(lon))

    # Use all time steps directly
    middle_steps = np.arange(0, len(time), dtype=int)
    middle_times = time[middle_steps]

    nco.createVariable('time', np.dtype('float32').char, ('time'))
    nco.createVariable('lon', np.dtype('float32').char, ('lon'))
    nco.createVariable('lat', np.dtype('float32').char, ('lat'))

    Nt = nco.dimensions['time'].size
    Ntm = len(middle_times)
    assert Nt == Ntm

    nco.variables['time'][:] = middle_times
    nco.variables['lon'][:] = lon
    nco.variables['lat'][:] = lat

    nco.close()
    nc.close()
