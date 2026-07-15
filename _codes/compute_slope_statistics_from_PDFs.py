"""
Compute Gram-Charlier slope statistics (c21, c03, c40, c04, c22) from
2-D slope PDFs for each DoLP gain condition; save to NetCDF.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import netCDF4 as nc

from subroutines.utils import *

import warnings

warnings.filterwarnings("ignore")

import os
# EPSS_STATS_SUFFIX: read timeseries_{gain}{SUFFIX}.nc and write
# slope_statistics_dataset{SUFFIX}.nc (default '' = published/canonical)
_SFX = os.environ.get('EPSS_STATS_SUFFIX', '')

path = '../_data/'
output_file_name = '../_data/slope_statistics_dataset%s.nc' % _SFX
pathname = Path(output_file_name)

# Inputs (including this script); recompute whenever any is newer than the output
input_files = [
    path + 'ASIT2019_wave_spectra_stats_timeseries_no_gain%s.nc' % _SFX,
    path + 'ASIT2019_wave_spectra_stats_timeseries_lab_gain%s.nc' % _SFX,
    path + 'ASIT2019_wave_spectra_stats_timeseries_empirical_gain%s.nc' % _SFX,
    path + 'ASIT2019_supporting_environmental_observations.nc',
    __file__,
]


def _output_up_to_date(out_path, in_paths):
    if not out_path.exists():
        return False
    out_mtime = out_path.stat().st_mtime
    return all(Path(p).stat().st_mtime <= out_mtime for p in in_paths)


if _output_up_to_date(pathname, input_files):
    print(f"File already up to date: {pathname}")

else:

    print("Computing wave slope statistics...")

    color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']

    kappa = 0.4
    
    ds_no = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_no_gain%s.nc' % _SFX)
    ds_lab = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_lab_gain%s.nc' % _SFX)
    ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain%s.nc' % _SFX)
    
    ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
    
    # Best 10-m wind speed (COARE, BUZM3 fallback) -> all 190 runs covered
    U10_m_s = np.ma.filled(ds_other["U10_best"][:].astype(float), np.nan)

    num_runs = len(U10_m_s)
    num_samples = ds_no["slope_east"].shape[1]

    # Wind-from direction for the upwind/crosswind rotation: COARE, with BUZM3
    # fallback where COARE is missing (mirrors the U10_best pairing)
    COARE_Wdir = np.ma.filled(ds_other["COARE_Wdir"][:].astype(float), np.nan)
    buzm3_Wdir = np.ma.filled(ds_other["buzm3_WDIR"][:].astype(float), np.nan)
    Wdir_best = np.where(np.isfinite(COARE_Wdir), COARE_Wdir, buzm3_Wdir)
    COARE_Wdir_vec = np.reshape(Wdir_best,(num_runs,1,1))

    slope_north = np.nan*np.ones((num_runs,num_samples,3))
    slope_east = np.nan*np.ones((num_runs,num_samples,3))
    
    slope_north[:,:,0] = ds_no["slope_north"][:]
    slope_north[:,:,1] = ds_lab["slope_north"][:]
    slope_north[:,:,2] = ds_emp["slope_north"][:]
    
    slope_east[:,:,0] = ds_no["slope_east"][:]
    slope_east[:,:,1] = ds_lab["slope_east"][:]
    slope_east[:,:,2] = ds_emp["slope_east"][:]
    
    slope_crosswind_long = np.cos(COARE_Wdir_vec*np.pi/180)*slope_east - np.sin(COARE_Wdir_vec*np.pi/180)*slope_north
    slope_upwind_long = np.sin(COARE_Wdir_vec*np.pi/180)*slope_east + np.cos(COARE_Wdir_vec*np.pi/180)*slope_north
    
    mss_long = np.nan*np.ones((num_runs,3,2))
    mss_long[:,:,0] = np.var(slope_crosswind_long,axis=1)
    mss_long[:,:,1] = np.var(slope_upwind_long,axis=1)
    
    # GC fits in the histograms' native upwind-positive frame (the C&M/B&H
    # sign convention; calibrated against the published third slope moments
    # 2026-07-06 -- a *-1 flip here would negate c21/c03 vs those references)
    slope_centers = ds_no["slope_centers"][:]
    slope_histogram_crosswind_upwind_no = ds_no["slope_histogram_crosswind_upwind"][:]
    slope_histogram_crosswind_upwind_lab = ds_lab["slope_histogram_crosswind_upwind"][:]
    slope_histogram_crosswind_upwind_emp = ds_emp["slope_histogram_crosswind_upwind"][:]
    
    mss_crosswind_no = ds_no["mss_crosswind"][:]
    mss_crosswind_lab = ds_lab["mss_crosswind"][:]
    mss_crosswind_emp = ds_emp["mss_crosswind"][:]
    
    mss_upwind_no = ds_no["mss_upwind"][:]
    mss_upwind_lab = ds_lab["mss_upwind"][:]
    mss_upwind_emp = ds_emp["mss_upwind"][:]
    
    mss = np.nan*np.ones((num_runs,3,2))
    
    mss[:,0,0] = mss_crosswind_no
    mss[:,1,0] = mss_crosswind_lab
    mss[:,2,0] = mss_crosswind_emp
    
    mss[:,0,1] = mss_upwind_no
    mss[:,1,1] = mss_upwind_lab
    mss[:,2,1] = mss_upwind_emp

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
    
    print("Done computing wave slope statistics!")
    
