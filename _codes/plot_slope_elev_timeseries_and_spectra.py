"""
Plot omnidirectional elevation spectra from the E-PSS slope-to-elevation
inversion, comparing E-PSS (no/lab/empirical gain) against the lidar reference.
(The slope/elevation timeseries demo now lives in the long-wave/short-wave
reconstruction figure, shown later in the manuscript.)
"""

import numpy as np

from scipy import signal

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

from subroutines.utils import figure_style, omni_complete_spectrum
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")

path = '../_data/'

ds_no = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_no_gain.nc')
ds_lab = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_lab_gain.nc')
ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')

elev_m_lidar = ds_other["wse_m_Riegl"][:]

slope_east_no = ds_no['slope_east'][:]
slope_north_no = ds_no['slope_north'][:]
slope_east_lab = ds_lab['slope_east'][:]
slope_north_lab = ds_lab['slope_north'][:]
slope_east_emp = ds_emp['slope_east'][:]
slope_north_emp = ds_emp['slope_north'][:]

sampling_rate_PSS = np.float64(10)
sampling_rate_lidar = np.float64(10)

nperseg = 1024
run_ind = 51
water_depth_m = 15.0

# Omnidirectional elevation spectra: lidar reference and the three E-PSS gains
f_Hz_lidar, Pxx_den_lidar = signal.welch(elev_m_lidar[0,:,run_ind], sampling_rate_lidar, nperseg=nperseg)
f_Hz, Pxx_den_no = omni_complete_spectrum(slope_east_no[run_ind], slope_north_no[run_ind], water_depth_m, sampling_rate_PSS, highpass_peak_fraction=0.5, nfft=nperseg, nperseg=nperseg)
f_Hz, Pxx_den_lab = omni_complete_spectrum(slope_east_lab[run_ind], slope_north_lab[run_ind], water_depth_m, sampling_rate_PSS, highpass_peak_fraction=0.5, nfft=nperseg, nperseg=nperseg)
f_Hz, Pxx_den_emp = omni_complete_spectrum(slope_east_emp[run_ind], slope_north_emp[run_ind], water_depth_m, sampling_rate_PSS, highpass_peak_fraction=0.5, nfft=nperseg, nperseg=nperseg)

i_lid = (f_Hz_lidar > 0.05) & (f_Hz_lidar < 1.0)
i_pss = (f_Hz > 0.05) & (f_Hz < 1.0)

fig = plt.figure(figsize=(fullwidth/2,fullwidth/2*4/3))

plt.plot(f_Hz_lidar[i_lid],Pxx_den_lidar[i_lid],color='black',linewidth=2,label="lidar")
plt.plot(f_Hz[i_pss],Pxx_den_no[i_pss],color=color_list[0],linewidth=2,alpha=0.75,label="E-PSS, no gain")
plt.plot(f_Hz[i_pss],Pxx_den_lab[i_pss],color=color_list[1],linewidth=2,alpha=0.75,label="E-PSS, lab gain")
plt.plot(f_Hz[i_pss],Pxx_den_emp[i_pss],color=color_list[2],linewidth=2,alpha=0.75,label="E-PSS, emp. gain")

plt.grid(which='major', linestyle='-', linewidth=0.75)
plt.grid(which='minor', linestyle=':', linewidth=0.75)

plt.xlim(1e-2,1e1)
plt.ylim(1e-6,1e0)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('f [Hz]')
plt.ylabel(r'$F_{\eta\eta}(f)$ [m$^2$Hz$^{-1}$]')

plt.legend(loc='lower left')

plt.savefig('../_figures/elevation_omnispect.pdf',bbox_inches='tight')
