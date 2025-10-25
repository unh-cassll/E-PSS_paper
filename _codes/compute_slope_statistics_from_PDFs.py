"""
Created on Mon Sep 15 22:24:47 2025

@author: nathanlaxague
"""

import sys

sys.path.append('subroutines/')
sys.path.append('../_codes/subroutines/')

from pathlib import Path

import numpy as np
import xarray as xr

import netCDF4 as nc

from utils import *

output_file_name = '../_data/slope_statistics_dataset.nc'
pathname = Path(output_file_name)

if pathname.exists():
    print(f"File already exists: {pathname}")
    
else:

    # Set custom property cycle colors
    color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']
    
    path = '../_data/'
    
    kappa = 0.4
    
    ds_no = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_no_gain.nc')
    ds_lab = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_lab_gain.nc')
    ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
    
    ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
    
    U_m_s = ds_other["EC_U_m_s"][:]
    ustar_m_s = ds_other["EC_ustar_m_s"][:]
    z_m_above_water = ds_other["EC_z_m_above_water"][:]
    
    U10_m_s = ustar_m_s/kappa*np.log10(10.0/z_m_above_water) + U_m_s
    
    COARE_Wdir_vec = np.reshape(ds_other["COARE_Wdir"][:],(190,1,1))
    
    slope_north = np.nan*np.ones((190,18000,3))
    slope_east = np.nan*np.ones((190,18000,3))
    
    slope_north[:,:,0] = ds_no["slope_north"][:]
    slope_north[:,:,1] = ds_lab["slope_north"][:]
    slope_north[:,:,2] = ds_emp["slope_north"][:]
    
    slope_east[:,:,0] = ds_no["slope_east"][:]
    slope_east[:,:,1] = ds_lab["slope_east"][:]
    slope_east[:,:,2] = ds_emp["slope_east"][:]
    
    slope_crosswind_long = np.cos(COARE_Wdir_vec*np.pi/180)*slope_east - np.sin(COARE_Wdir_vec*np.pi/180)*slope_north
    slope_upwind_long = np.sin(COARE_Wdir_vec*np.pi/180)*slope_east + np.cos(COARE_Wdir_vec*np.pi/180)*slope_north
    
    mss_long = np.nan*np.ones((190,3,2))
    mss_long[:,:,0] = np.var(slope_crosswind_long,axis=1)
    mss_long[:,:,1] = np.var(slope_upwind_long,axis=1)
    
    slope_centers = ds_no["slope_centers"][:]*-1
    slope_histogram_crosswind_upwind_no = ds_no["slope_histogram_crosswind_upwind"][:]
    slope_histogram_crosswind_upwind_lab = ds_lab["slope_histogram_crosswind_upwind"][:]
    slope_histogram_crosswind_upwind_emp = ds_emp["slope_histogram_crosswind_upwind"][:]
    
    mss_crosswind_no = ds_no["mss_crosswind"][:]
    mss_crosswind_lab = ds_lab["mss_crosswind"][:]
    mss_crosswind_emp = ds_emp["mss_crosswind"][:]
    
    mss_upwind_no = ds_no["mss_upwind"][:]
    mss_upwind_lab = ds_lab["mss_upwind"][:]
    mss_upwind_emp = ds_emp["mss_upwind"][:]
    
    mss = np.nan*np.ones((190,3,2))
    
    mss[:,0,0] = mss_crosswind_no
    mss[:,1,0] = mss_crosswind_lab
    mss[:,2,0] = mss_crosswind_emp
    
    mss[:,0,1] = mss_upwind_no
    mss[:,1,1] = mss_upwind_lab
    mss[:,2,1] = mss_upwind_emp
    
    num_runs = len(U10_m_s)
    
    slope_stats_array = np.nan*np.ones((num_runs,7,3))
    slope_stats_output_names = ['c21','c03','c40','c04','c22','R_squared','RMSE']
    
    for i in range(len(U10_m_s)):
        
        out_struc_no = fit_gram_charlier_slope_pdf(slope_centers, slope_histogram_crosswind_upwind_no[i,:,:], mss_upwind_no[i], mss_crosswind_no[i])
        out_struc_lab = fit_gram_charlier_slope_pdf(slope_centers, slope_histogram_crosswind_upwind_lab[i,:,:], mss_upwind_lab[i], mss_crosswind_lab[i])        
        out_struc_emp = fit_gram_charlier_slope_pdf(slope_centers, slope_histogram_crosswind_upwind_emp[i,:,:], mss_upwind_emp[i], mss_crosswind_emp[i])    
        
        for j in range(7):
            
            slope_stats_array[i,j,0] = out_struc_no[slope_stats_output_names[j]]
            slope_stats_array[i,j,1] = out_struc_lab[slope_stats_output_names[j]]
            slope_stats_array[i,j,2] = out_struc_emp[slope_stats_output_names[j]]
    
    category = np.concatenate((np.full(num_runs, 'no gain'),np.full(num_runs, 'lab gain'),np.full(num_runs, 'empirical gain')))
    color = np.concatenate((np.full(num_runs, color_list[0]),np.full(num_runs, color_list[1]),np.full(num_runs, color_list[2])))
    
    slope_stats_ds = xr.Dataset()
    slope_stats_ds['U10'] = np.concatenate((U10_m_s,U10_m_s,U10_m_s))
    
    slope_stats_ds['mss_u'] = np.concatenate((mss[:,0,1],mss[:,1,1],mss[:,2,1]))
    slope_stats_ds['mss_c'] = np.concatenate((mss[:,0,0],mss[:,1,0],mss[:,2,0]))
    
    slope_stats_ds['mss_u_long'] = np.concatenate((mss_long[:,0,1],mss_long[:,1,1],mss_long[:,2,1]))
    slope_stats_ds['mss_c_long'] = np.concatenate((mss_long[:,0,0],mss_long[:,1,0],mss_long[:,2,0]))
    
    slope_stats_ds['c21'] = np.concatenate((slope_stats_array[:,0,0],slope_stats_array[:,0,1],slope_stats_array[:,0,2]))
    slope_stats_ds['c03'] = np.concatenate((slope_stats_array[:,1,0],slope_stats_array[:,1,1],slope_stats_array[:,1,2]))
    slope_stats_ds['c40'] = np.concatenate((slope_stats_array[:,2,0],slope_stats_array[:,2,1],slope_stats_array[:,2,2]))
    slope_stats_ds['c04'] = np.concatenate((slope_stats_array[:,3,0],slope_stats_array[:,3,1],slope_stats_array[:,3,2]))
    slope_stats_ds['c22'] = np.concatenate((slope_stats_array[:,4,0],slope_stats_array[:,4,1],slope_stats_array[:,4,2]))
    
    slope_stats_ds['R_squared'] = np.concatenate((slope_stats_array[:,5,0],slope_stats_array[:,5,1],slope_stats_array[:,5,2]))
    slope_stats_ds['RMSE'] = np.concatenate((slope_stats_array[:,6,0],slope_stats_array[:,6,1],slope_stats_array[:,6,2]))
    
    slope_stats_ds['DoLP gain'] = category
    slope_stats_ds['color'] = color
    
    slope_stats_ds["U10"].attrs = {"units": "m/s", "long_name": "ten-meter wind speed"}
    
    slope_stats_ds["mss_u"].attrs = {"units": "none", "long_name": "upwind slope variance"}
    slope_stats_ds["mss_c"].attrs = {"units": "none", "long_name": "crosswind slope variance"}
    
    slope_stats_ds["mss_u_long"].attrs = {"units": "none", "long_name": "upwind slope variance, long waves"}
    slope_stats_ds["mss_c_long"].attrs = {"units": "none", "long_name": "crosswind slope variance, long waves"}
    
    slope_stats_ds["c21"].attrs = {"units": "none", "long_name": "Gram-Charlier coefficient c21 (~skewness)"}
    slope_stats_ds["c03"].attrs = {"units": "none", "long_name": "Gram-Charlier coefficient c03 (~skewness)"}
    slope_stats_ds["c40"].attrs = {"units": "none", "long_name": "Gram-Charlier coefficient c40 (~kurtosis)"}
    slope_stats_ds["c04"].attrs = {"units": "none", "long_name": "Gram-Charlier coefficient c04 (~kurtosis)"}
    slope_stats_ds["c22"].attrs = {"units": "none", "long_name": "Gram-Charlier coefficient c22 (~kurtosis)"}
    
    slope_stats_ds["R_squared"].attrs = {"units": "none", "long_name": "coefficient of determination"}
    slope_stats_ds["RMSE"].attrs = {"units": "none", "long_name": "root-mean-square-error"}
    
    slope_stats_ds["DoLP gain"].attrs = {"units": "none", "long_name": "DoLP gain category"}
    slope_stats_ds["color"].attrs = {"units": "none", "long_name": "color for plotting"}

    slope_stats_ds.to_netcdf(output_file_name)
    
    