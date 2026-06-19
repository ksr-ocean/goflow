#!/usr/bin/env python

import netCDF4 as nc
import numpy as np
from tqdm import tqdm
import sys
import glob
def kelvin_to_celsius(kelvin_array):
    """Converts a NumPy array from Kelvin to Celsius."""
    return kelvin_array.astype(np.float32) - 273.15

def calculate_distances(lats, subsampled_lons):
    """
    Calculate cumulative distances in kilometers along latitude and longitude arrays.
    """
    R = 6371.0  # Earth radius in kilometers
    lons_rad = np.deg2rad(subsampled_lons)
    lats_rad = np.deg2rad(lats)
    lat_dist_per_degree = np.pi * R / 180.0
    
    # Cumulative distance along latitude
    lat_distances_array = np.cumsum(np.abs(np.diff(lats))) * lat_dist_per_degree
    lat_distances_array = np.insert(lat_distances_array, 0, 0)

    # Cumulative distance along longitude
    mean_lat_rad = np.mean(lats_rad)
    lon_dist_per_degree = lat_dist_per_degree * np.cos(mean_lat_rad)
    lon_distances_array = np.cumsum(np.abs(np.diff(subsampled_lons))) * lon_dist_per_degree
    lon_distances_array = np.insert(lon_distances_array, 0, 0)

    return lat_distances_array, lon_distances_array

def copy_attributes(src_var, dst_var):
    """
    Copies all attributes from a source NetCDF variable to a destination variable.
    """
    for attr_name in src_var.ncattrs():
        attr_value = src_var.getncattr(attr_name)
        dst_var.setncattr(attr_name, attr_value)
    print(f"Copied {len(src_var.ncattrs())} attributes for variable '{dst_var.name}'.")

def modNcfile(fin, maskfile, t0=0, step=1):
    """
    Processes a NetCDF file to calculate temperature gradients, apply a mask,
    and save the output to a new file, preserving coordinate attributes.
    """
    fo = fin.split('.')[0] + '_grad_mask.nc'
    print(f"Input file: {fin}")
    print(f"Mask file: {maskfile}")
    print(f"Output file: {fo}")

    try:
        ncm = nc.Dataset(maskfile, 'r')
        with nc.Dataset(fin, 'r') as src_ds:
            # Read dimensions and coordinate variables
            times = src_ds.variables['time'][:]
            original_lons = src_ds.variables['longitude'][::2]
            original_lats = src_ds.variables['latitude'][::2]
            
            lat_dist_km, lon_dist_km = calculate_distances(original_lats, original_lons)
            
            # Create a new NetCDF file for the processed data
            with nc.Dataset(fo, 'w', format='NETCDF4') as dst_ds:
                # --- Create Dimensions ---
                dst_ds.createDimension('time', None)
                dst_ds.createDimension('lat', len(original_lats))
                dst_ds.createDimension('lon', len(original_lons))

                # --- Create Variables ---
                times_var = dst_ds.createVariable('time', 'f8', ('time',))
                lats_var = dst_ds.createVariable('lat', 'f8', ('lat',))
                lons_var = dst_ds.createVariable('lon', 'f8', ('lon',))
                
                bt_var = dst_ds.createVariable('BT', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan)
                gradT_var = dst_ds.createVariable('gradT', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan)
                log_gradT_var = dst_ds.createVariable('log_gradT', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan)
                mask_var = dst_ds.createVariable('mask', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan)
                log_gradT_masked_var = dst_ds.createVariable('log_gradT_masked', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan)

                # --- Set Variable Attributes ---
                bt_var.long_name = "Brightness Temperature, ABI Band 14"
                bt_var.standard_name = "BT"
                bt_var.units = "Celsius"
                bt_var.comment = "Brightness Temperature, ABI Band 14 GOES 16, converted to Celsius and axes flipped."
                gradT_var.long_name = "Magnitude of Brightness Temperature Gradient"
                gradT_var.units = "Celsius / m"
                
                # --- Copy Attributes from Source (Integrated Logic) ---
                copy_attributes(src_ds.variables['time'], times_var)
                copy_attributes(src_ds.variables['latitude'], lats_var)
                copy_attributes(src_ds.variables['longitude'], lons_var)

                # --- Write Coordinate Data ---
                times_var[:] = times[t0::step]
                lats_var[:] = original_lats
                lons_var[:] = original_lons

                # --- Process and Write Data Time Slice by Time Slice ---
                for it, t in enumerate(tqdm(range(t0, len(times), step))):
                    bt_slice = src_ds.variables['BT'][t, ::2, ::2].astype(np.float32)
                    mask_slice = ncm.variables['MASK'][t, ::2, ::2].astype(np.float32)
                    
                    bt_slice[bt_slice == -999] = np.nan
                    mask_slice[np.isnan(mask_slice)] = 0
                    
                    bt_slice_celsius = kelvin_to_celsius(bt_slice)
                    
                    # Flip lat/lon axes to match convention
                    bt_slice_flipped = np.swapaxes(bt_slice_celsius, 0, 1)
                    mask_flipped = np.swapaxes(mask_slice, 0, 1)
                    
                    bt_var[it, :, :] = np.nan_to_num(bt_slice_flipped)
                    
                    # Calculate gradient in Celsius per meter
                    grad_lat, grad_lon = np.gradient(bt_slice_flipped, lat_dist_km * 1000, lon_dist_km * 1000, edge_order=2)
                    gradTVal = (grad_lat**2 + grad_lon**2)**0.5
                    
                    # Write processed data
                    gradT_var[it, :, :] =  np.nan_to_num(gradTVal)
                    log_gradT_var[it, :, :] =  np.nan_to_num(np.log10(gradTVal))
                    log_gradT_masked_var[it, :, :] =  np.nan_to_num(np.log10(gradTVal) * mask_flipped)
                    mask_var[it, :, :] =  np.nan_to_num(mask_flipped)
                    
    except FileNotFoundError:
        print(f"Error: One or both input files not found. Please check paths.")
        print(f" - Data file: {fin}")
        print(f" - Mask file: {maskfile}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: A variable was not found in one of the NetCDF files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        if 'ncm' in locals() and ncm.isopen():
            ncm.close()

if __name__ == "__main__":
    #input_file = sys.argv[1]
    #mask_file = sys.argv[2]
    files = glob.glob('GS_2022*V2.nc'); files.sort()
    #files = ['GS_20231101T000000_20231201T000000_DT01_V2.nc','GS_20231001T000000_20231101T000000_DT01_V2.nc' ]
    files_mask = [f.split('.')[0] + '_cloudmask.nc' for f in files]
    #files_mask = ['GS_20230501T000000_20230601T000000_DT01_V2_cloudmask.nc'] 
    for f, m in zip(files, files_mask):
        modNcfile(f, m)

