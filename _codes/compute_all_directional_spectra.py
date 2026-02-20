"""
Created on Wed Sep 17 07:58:12 2025

@author: nathanlaxague
"""

from pathlib import Path

import numpy as np
import xarray as xr

import netCDF4 as nc

from subroutines.utils import *

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

output_file_name = '../_data/ASIT2019_EPSS_directional_spectra.nc'
pathname = Path(output_file_name)

if pathname.exists():
    print(f"File already exists: {pathname}")
    
else:
    
    print("Computing frequency-directional spectra via E-PSS/MEM...")

    g = 9.81
    
    path = '../_data/'
    
    fn = path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc'
    ds = nc.Dataset(fn)
    
    ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
        
    slope_north = ds['slope_north'][:]
    slope_east = ds['slope_east'][:]
    
    f_Hz = ds['f_Hz'][:]
    
    fs_Hz = np.floor(2*f_Hz[len(f_Hz)-1])
    
    f_Hz_Pyxis = ds['f_Hz'][:]
    theta_rad_Pyxis = ds['theta_rad'][:]
    S_f_theta_Pyxis = ds['S_f_theta'][:]
                
    num_samples = len(slope_north[1,:])
    nfft = num_samples/10
    nperseg = nfft/2
    
    num_runs = 190
    num_f = np.int16(nfft/2+1)
    
    sampling_rate_PSS = 30
    f_lp = 1/2
    f_hp = 1/15
    
    spect_low_f = 0.05
    spect_high_f = 1
    
    water_depth_m = 15.0
    
    smoothnum = 3
    
    theta_halfwidth = 90
    
    fmin = 5e-2
    fmax = 6e-1
    
    f_cut_spect = 0.5
    f_cut_high = 0.35
    
    num_dirs = 72
    
    F_EPSS_stack = np.nan*np.ones((num_f,num_dirs,num_runs))
    
    for run_ind in np.arange(0,num_runs):
    
        S_f_theta_Pyxis_particular = S_f_theta_Pyxis[run_ind,:,:]
        S_theta = np.sum(S_f_theta_Pyxis_particular,axis=1)
        ind = np.argmax(S_theta)
        short_mwd = 180/np.pi*theta_rad_Pyxis[ind]
        
        k_disp = (2*np.pi*f_Hz_Pyxis)**2/g
        k_disp_mat = np.tile(k_disp,(len(theta_rad_Pyxis),1))
        Fftheta_m2_Hz_rad_Pyxis = (S_f_theta_Pyxis_particular*k_disp_mat**-2).T
    
        Fftheta_m2_Hz_rad_Pyxis_shifted = np.concatenate((Fftheta_m2_Hz_rad_Pyxis[:,np.arange(36,72)],Fftheta_m2_Hz_rad_Pyxis[:,np.arange(0,36)]),axis=1)
        theta_rad_Pyxis_shifted = np.concatenate((theta_rad_Pyxis[np.arange(36,72)]-2*np.pi,theta_rad_Pyxis[np.arange(0,36)]))
    
        # creating dataset (Pyxis frequency spectrum)
        dataset_Pyxis_frequency = xr.Dataset(
            coords = {"frequency": f_Hz_Pyxis, "direction": 180/np.pi*theta_rad_Pyxis_shifted},
            data_vars = {
                "Ffd": (["frequency", "direction"], Fftheta_m2_Hz_rad_Pyxis_shifted*np.pi/180)
            }
        )
        Ffd_direct = dataset_Pyxis_frequency.Ffd
        
        sE = slope_east[run_ind,:]
        sN = slope_north[run_ind,:]
        
        sE = np.where(np.isfinite(sE), sE, 0.0)
        sN = np.where(np.isfinite(sN), sN, 0.0)
        
        elev_m = slope_to_elev(sE,sN,water_depth_m,1/sampling_rate_PSS,f_lp,f_hp)
                
        F_EPSS = compute_dirspec_EPSS(elev_m,sE,sN,fs_Hz,spect_low_f,spect_high_f,nfft,nperseg,smoothnum)
        F_EPSS = trim_EPSS_dirspec(F_EPSS,theta_halfwidth,fmin,fmax,smoothnum)
        
        F_EPSS = F_EPSS*np.var(elev_m)/F_EPSS.integrate('frequency').integrate('direction')
        
        F_EPSS_stack[:,:,run_ind] = F_EPSS.data
        
        

    F_EPSS_ds = xr.Dataset(
        coords = {"frequency": F_EPSS['frequency'].data, "direction": F_EPSS['direction'].data, "run number": np.arange(num_runs)},
        data_vars = {
            "F_f_d": (["frequency", "direction", "run"], F_EPSS_stack)
        },
        attrs={'units': 'm^2/Hz/deg'}
    )
            
    F_EPSS_ds['frequency'].attrs = {'units': 'Hz'}
    F_EPSS_ds['direction'].attrs = {'units': 'degrees clockwise from true North'}
    F_EPSS_ds['run number'].attrs = {'units': 'sequential run number'}
    
    F_EPSS_ds.to_netcdf(output_file_name)
    
    print("Done computing frequency-directional spectra via E-PSS/MEM!")

