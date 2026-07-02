"""
Plot omnidirectional elevation spectra from the E-PSS slope-to-elevation
inversion, comparing E-PSS (no/lab/empirical gain) against the lidar reference.
"""

import numpy as np

from scipy import signal

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

from subroutines.utils import (figure_style, omni_complete_spectrum, L_FOV_M,
                               WATER_DEPTH_M, FS_HZ)
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

sampling_rate_PSS = FS_HZ
sampling_rate_lidar = FS_HZ

nperseg = 1024
run_ind = 51
water_depth_m = WATER_DEPTH_M

# Omnidirectional elevation spectra: lidar reference and the three E-PSS gains
f_Hz_lidar, Pxx_den_lidar = signal.welch(elev_m_lidar[0,:,run_ind], sampling_rate_lidar, nperseg=nperseg)
f_Hz, Pxx_den_no = omni_complete_spectrum(slope_east_no[run_ind], slope_north_no[run_ind], water_depth_m, sampling_rate_PSS, highpass_peak_fraction=0.5, nfft=nperseg, nperseg=nperseg, aperture_diameter_m=L_FOV_M)
f_Hz, Pxx_den_lab = omni_complete_spectrum(slope_east_lab[run_ind], slope_north_lab[run_ind], water_depth_m, sampling_rate_PSS, highpass_peak_fraction=0.5, nfft=nperseg, nperseg=nperseg, aperture_diameter_m=L_FOV_M)
f_Hz, Pxx_den_emp = omni_complete_spectrum(slope_east_emp[run_ind], slope_north_emp[run_ind], water_depth_m, sampling_rate_PSS, highpass_peak_fraction=0.5, nfft=nperseg, nperseg=nperseg, aperture_diameter_m=L_FOV_M)

# Display 0.05-5 Hz; fade each spectrum outside the validated passband [f_hp, f_lp]
f_hp, f_lp = 0.08, 0.5
fmin_disp, fmax_disp = 0.05, 5.0
alpha_faded = 0.30

fig = plt.figure(figsize=(fullwidth/2,fullwidth/2*4/3))

def plot_passband(fx, S, color, label, full_alpha):
    disp = (fx >= fmin_disp) & (fx <= fmax_disp)
    inb = disp & (fx >= f_hp) & (fx <= f_lp)
    # continuous faded baseline over the whole display range (no gap at the
    # cutoffs), then the in-band segment overlaid at full opacity
    plt.plot(fx[disp], S[disp], color=color, linewidth=2, alpha=alpha_faded)
    plt.plot(fx[inb], S[inb], color=color, linewidth=2, alpha=full_alpha, label=label)

plot_passband(f_Hz_lidar, Pxx_den_lidar, 'black', 'lidar', 1.0)
plot_passband(f_Hz, Pxx_den_no, color_list[0], 'E-PSS, no gain', 0.75)
plot_passband(f_Hz, Pxx_den_lab, color_list[1], 'E-PSS, lab gain', 0.75)
plot_passband(f_Hz, Pxx_den_emp, color_list[2], 'E-PSS, emp. gain', 0.75)
for b in (f_hp, f_lp):
    plt.axvline(b, color='dimgray', linestyle='--', linewidth=1, alpha=0.7)

plt.grid(which='major', linestyle='-', linewidth=0.75)
plt.grid(which='minor', linestyle=':', linewidth=0.75)

plt.xlim(fmin_disp, fmax_disp)
plt.ylim(1e-5,1e0)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('f [Hz]')
plt.ylabel(r'$F_{\eta\eta}(f)$ [m$^2$Hz$^{-1}$]')

plt.legend(loc='lower left')

plt.savefig('../_figures/elevation_omnispect.pdf',bbox_inches='tight')
