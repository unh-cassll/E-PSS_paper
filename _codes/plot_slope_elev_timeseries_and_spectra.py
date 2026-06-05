"""
Plot wave slope and elevation timeseries and omnidirectional elevation spectra.
Compares E-PSS (no/lab/empirical gain) against lidar reference.
"""

import numpy as np

from scipy import signal

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

from subroutines.utils import figure_style, slope_to_elev_wavelet, omni_complete_spectrum
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")

g = 9.81;

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

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')

sampling_rate_PSS = np.float64(10)
sampling_rate_lidar = np.float64(10)

nperseg = 1024
run_ind = 51

dt = 1/sampling_rate_PSS
t = np.arange(slope_east_emp.shape[1]) * dt

water_depth_m = 15.0
f_lp = 1/2
f_hp = 0.08   # Hz; highpass corner for elevation inversion; suppresses 1/k²-amplified low-f drift

sE = slope_east_no[run_ind,:]
sN = slope_north_no[run_ind,:]
sE = np.where(np.isfinite(sE), sE, 0.0)
sN = np.where(np.isfinite(sN), sN, 0.0)
elev_m_no = slope_to_elev_wavelet(sE,sN,water_depth_m,sampling_rate_PSS,fmin_Hz=f_hp)

sE = slope_east_lab[run_ind,:]
sN = slope_north_lab[run_ind,:]
sE = np.where(np.isfinite(sE), sE, 0.0)
sN = np.where(np.isfinite(sN), sN, 0.0)
elev_m_lab = slope_to_elev_wavelet(sE,sN,water_depth_m,sampling_rate_PSS,fmin_Hz=f_hp)

sE = slope_east_emp[run_ind,:]
sN = slope_north_emp[run_ind,:]
sE = np.where(np.isfinite(sE), sE, 0.0)
sN = np.where(np.isfinite(sN), sN, 0.0)
elev_m_emp = slope_to_elev_wavelet(sE,sN,water_depth_m,sampling_rate_PSS,fmin_Hz=f_hp)

slope_lim_val = 0.3

t_start = 456   # s; start of displayed 20-s window

t = t - t_start

fig, ax1 = plt.subplots(figsize=(fullwidth, fullwidth/2))

plt.xlim(0,20)

# Demeaned slope timeseries (removes fixed viewing-angle tilt)
sE_plot = slope_east_emp[run_ind,:] - np.mean(slope_east_emp[run_ind,:])
sN_plot = slope_north_emp[run_ind,:] - np.mean(slope_north_emp[run_ind,:])
ax1.plot(t,sE_plot, '--',linewidth=2, label='$s_E$', color='black')
ax1.plot(t,sN_plot, ':',linewidth=2, label='$s_N$', color='black')
ax1.set_ylabel('slope [rad]')
ax1.tick_params(axis='y')
ax1.set_xlabel('t [s]')

ax1.set_ylim(-slope_lim_val,slope_lim_val)

ax1.legend(loc='upper left')

ax2 = ax1.twinx()

# E-PSS wavelet elevation timeseries; m
ax2.plot(t, elev_m_emp, linewidth=2, color=color_list[2])
ax2.set_ylabel('$\eta$ [m]',color=color_list[2])
ax2.tick_params(axis='y',labelcolor=color_list[2])

ax2.set_ylim(-2*slope_lim_val,2*slope_lim_val)

plt.savefig('../_figures/wave_slope_elev_timeseries.pdf',bbox_inches='tight')

# %%

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

print(f'[run {run_ind}] E-PSS long-wave eta: std={np.std(elev_m_emp):.3f} m')

