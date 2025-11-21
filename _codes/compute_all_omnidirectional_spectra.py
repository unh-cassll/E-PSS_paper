"""
Created on Sat Nov 15 22:47:48 2025

@author: nathanlaxague
"""

import numpy as np
import xarray as xr

import netCDF4 as nc

from scipy import signal

from subroutines.utils import *

from pathlib import Path

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

output_file_name = '../_data/ASIT2019_omnidirectional_spectra.nc'
pathname = Path(output_file_name)

if pathname.exists():
    print(f"File already exists: {pathname}")
    
else:
    
    print("Computing omnidirectional frequency spectra...")
    
    path = '../_data/'
    
    ds_no = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_no_gain.nc')
    ds_lab = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_lab_gain.nc')
    ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
    
    slope_east_no = ds_no['slope_east'][:]
    slope_north_no = ds_no['slope_north'][:]
    
    slope_east_lab = ds_lab['slope_east'][:]
    slope_north_lab = ds_lab['slope_north'][:]
    
    slope_east_emp = ds_emp['slope_east'][:]
    slope_north_emp = ds_emp['slope_north'][:]
    
    ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
    
    elev_m_lidar = ds_other["wse_m_Riegl"][:]

    nfft = 3000
    nperseg = 1500
    num_freqs = np.int16(nfft/2+1)
    num_runs = np.size(elev_m_lidar,axis=2)
    sampling_rate_PSS = 30
    sampling_rate_lidar = 10
    subnum = np.int16(sampling_rate_PSS/sampling_rate_lidar)
    num_lidars = 3
    
    run_number = np.arange(num_runs)
    
    water_depth_m = 15.0
    f_lp = 1/2
    f_hp = 1/12
    
    F_f_m2_Hz_lidar = np.nan*np.ones((num_freqs,num_runs,num_lidars))
    
    F_f_m2_Hz_no_gain = np.nan*np.ones((num_freqs,num_runs))
    F_f_m2_Hz_lab_gain = np.nan*np.ones((num_freqs,num_runs))
    F_f_m2_Hz_empirical_gain = np.nan*np.ones((num_freqs,num_runs))
    
    for run_ind in range(num_runs):
        
        sE = slope_east_no[run_ind,:]
        sN = slope_north_no[run_ind,:]    
        sE = np.where(np.isfinite(sE), sE, 0.0)
        sN = np.where(np.isfinite(sN), sN, 0.0)
        elev_m_no = slope_to_elev(sE,sN,water_depth_m,1/sampling_rate_PSS,f_lp,f_hp)
        elev_m_no = elev_m_no[::subnum]
        
        sE = slope_east_lab[run_ind,:]
        sN = slope_north_lab[run_ind,:]    
        sE = np.where(np.isfinite(sE), sE, 0.0)
        sN = np.where(np.isfinite(sN), sN, 0.0)
        elev_m_lab = slope_to_elev(sE,sN,water_depth_m,1/sampling_rate_PSS,f_lp,f_hp)
        elev_m_lab = elev_m_lab[::subnum]
        
        sE = slope_east_emp[run_ind,:]
        sN = slope_north_emp[run_ind,:]    
        sE = np.where(np.isfinite(sE), sE, 0.0)
        sN = np.where(np.isfinite(sN), sN, 0.0)
        elev_m_emp = slope_to_elev(sE,sN,water_depth_m,1/sampling_rate_PSS,f_lp,f_hp)
        elev_m_emp = elev_m_emp[::subnum]
        
        f_Hz, Pxx_den = signal.welch(elev_m_no, sampling_rate_lidar, nfft=nfft, nperseg=nperseg)
        F_f_m2_Hz_no_gain[:,run_ind] = Pxx_den
        
        f_Hz, Pxx_den = signal.welch(elev_m_lab, sampling_rate_lidar, nfft=nfft, nperseg=nperseg)
        F_f_m2_Hz_lab_gain[:,run_ind] = Pxx_den
        
        f_Hz, Pxx_den = signal.welch(elev_m_emp, sampling_rate_lidar, nfft=nfft, nperseg=nperseg)
        F_f_m2_Hz_empirical_gain[:,run_ind] = Pxx_den
        
        for lidar_ind in range(num_lidars):
            
            f_Hz, Pxx_den = signal.welch(elev_m_lidar[lidar_ind,:,run_ind], sampling_rate_lidar, nfft=nfft, nperseg=nperseg)
            F_f_m2_Hz_lidar[:,run_ind,lidar_ind] = Pxx_den    
            
    F_f_m2_Hz_lidar = np.median(F_f_m2_Hz_lidar,axis=2)
    
    F_f_m2_Hz_all_ds = xr.Dataset(
        {
            'F_f_m2_Hz_no_gain': (['frequency', 'run number'], F_f_m2_Hz_no_gain),
            'F_f_m2_Hz_lab_gain': (['frequency', 'run number'], F_f_m2_Hz_lab_gain),
            'F_f_m2_Hz_empirical_gain': (['frequency', 'run number'], F_f_m2_Hz_empirical_gain),
            'F_f_m2_Hz_lidar': (['frequency', 'run number'], F_f_m2_Hz_lidar),
        },
        coords={
            'frequency': f_Hz,
            'run number': run_number
        },
        attrs={'units': 'm^2/Hz'}
    )

    F_f_m2_Hz_all_ds['frequency'].attrs = {'units': 'Hz'}
    F_f_m2_Hz_all_ds['run number'].attrs = {'units': 'sequential run number'}
    
    F_f_m2_Hz_all_ds.to_netcdf(output_file_name)
    
    print("Done computing omnidirectional frequency spectra!")