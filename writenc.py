"""
NetCDF Output Utilities
=======================
Helper functions for creating and writing NetCDF output files.

Uses 'lon'/'lat' dimension naming convention (vs 'x'/'y' in utils.py).

"""

import numpy as np
from netCDF4 import Dataset


def ncCreate(fname: str, Nx: int, Ny: int, varlist: list, dt: float = None):
    nco = Dataset(fname, 'w')
    nco.createDimension('time', None)
    nco.createDimension('lon', Nx)
    nco.createDimension('lat', Ny)
    for var in varlist:
        nco.createVariable(var, np.dtype('float32').char, ('time', 'lat', 'lon'))
    if dt is not None:
        nco.dt = dt
    return nco 

def addVal(nco, var, val, itime):
    nco.variables[var][itime, :,:] = val.squeeze() 
        
