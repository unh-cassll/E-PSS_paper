"""
Created on Tue Sep  9 15:01:17 2025

@author: nathanlaxague
"""

import sys
sys.path.append('subroutines/')

import numpy as np

from scipy import signal

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

g = 9.81;

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

path = '../_data/'

ds_no = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_no_gain.nc')
ds_lab = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_lab_gain.nc')
ds_empirical = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')

elev_m_lidar = ds_other["wse_m_Riegl"][:]

elev_m_no = ds_no["elev_m"][:]
elev_m_lab = ds_lab["elev_m"][:]
elev_m_emp = ds_empirical["elev_m"][:]

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')

sampling_rate_PSS = np.float64(30)
sampling_rate_lidar = np.float64(10)

nperseg = 2048
    
elev_m = ds_empirical['elev_m'][:]
slope_north = ds_empirical['slope_north'][:]
slope_east = ds_empirical['slope_east'][:]

run_ind = 172

dt = 1/sampling_rate_PSS
t = np.arange(0,600,dt)

slope_lim_val = 0.3

t_start = 60

t = t - t_start

fig, ax1 = plt.subplots(figsize=(12, 6))

plt.xlim(0,20)

# slope timeseries
ax1.plot(t,slope_east[run_ind,:], '--',linewidth=2, label='$s_E$', color='black')
ax1.plot(t,slope_north[run_ind,:], ':',linewidth=2, label='$s_N$', color='black')
ax1.set_ylabel('slope [rad]')
ax1.tick_params(axis='y')
ax1.set_xlabel('t [s]')

ax1.set_ylim(-slope_lim_val,slope_lim_val)

ax1.legend(loc='upper left')

ax2 = ax1.twinx()

# elevation timeseries
ax2.plot(t,elev_m[run_ind,:],linewidth=2, label='r$\eta$', color=color_list[2])
ax2.set_ylabel('$\eta$ [m]',color=color_list[2])
ax2.tick_params(axis='y',labelcolor=color_list[2])

ax2.set_ylim(-2*slope_lim_val,2*slope_lim_val)

plt.savefig('../_figures/wave_slope_elev_timeseries.pdf',bbox_inches='tight')

# %%


f_Hz_lidar, Pxx_den_lidar = signal.welch(elev_m_lidar[0,:,run_ind], sampling_rate_lidar, nperseg=nperseg/3)
f_Hz, Pxx_den_no = signal.welch(elev_m_no[run_ind,:], sampling_rate_PSS, nperseg=nperseg)
f_Hz, Pxx_den_lab = signal.welch(elev_m_lab[run_ind,:], sampling_rate_PSS, nperseg=nperseg)
f_Hz, Pxx_den_emp = signal.welch(elev_m_emp[run_ind,:], sampling_rate_PSS, nperseg=nperseg)

fig = plt.figure(figsize=(6,5.5))

plt.plot(f_Hz_lidar[2:len(f_Hz_lidar)],Pxx_den_lidar[2:len(f_Hz_lidar)],color='black',linewidth=2,label="lidar")
plt.plot(f_Hz[2:len(f_Hz)],Pxx_den_no[2:len(f_Hz)],color=color_list[0],linewidth=2,alpha=0.75,label="E-PSS, no gain")
plt.plot(f_Hz[2:len(f_Hz)],Pxx_den_lab[2:len(f_Hz)],color=color_list[1],linewidth=2,alpha=0.75,label="E-PSS, lab gain")
plt.plot(f_Hz[2:len(f_Hz)],Pxx_den_emp[2:len(f_Hz)],color=color_list[2],linewidth=2,alpha=0.75,label="E-PSS, emp. gain")

plt.grid(which='major', linestyle='-', linewidth=0.75)
plt.grid(which='minor', linestyle=':', linewidth=0.75)

plt.xlim(2.5e-2,2.5e0)
plt.ylim(2e-4,2e0)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('f [Hz]')
plt.ylabel(r'$F_{\eta\eta}(f)$ [m$^2$Hz$^{-1}$]')

plt.legend()

plt.savefig('../_figures/elevation_omnispect.pdf',bbox_inches='tight')

