"""
Created on Tue Sep  9 15:01:17 2025

@author: nathanlaxague
"""

import numpy as np

from scipy import signal

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

from subroutines.utils import figure_style, slope_to_elev_wavelet, omni_complete_spectrum
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

# Suppress all warnings
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
f_hp = 0.08   # gentle highpass corner on the elevation inversion (matches the
              # legacy FFT pipeline); suppresses the 1/k^2-amplified low-f drift

# Empirical aperture-MTF curve (calibrated camera-only from the field's
# center-vs-mean slope spectra by make_aperture_mtf_curve.py); deconvolves the
# 0.3-0.7 Hz attenuation the footprint averaging imposes on the spatial-mean
# slope. These demonstration plots use the long-wave wavelet inversion only.
_mtf = np.load(path+'ASIT2019_aperture_mtf_gain.npz')
mtf_curve = (_mtf['freqs_Hz'], _mtf['gain'])

sE = slope_east_no[run_ind,:]
sN = slope_north_no[run_ind,:]
sE = np.where(np.isfinite(sE), sE, 0.0)
sN = np.where(np.isfinite(sN), sN, 0.0)
elev_m_no = slope_to_elev_wavelet(sE,sN,water_depth_m,sampling_rate_PSS,aperture_mtf_curve=mtf_curve,fmin_Hz=f_hp)

sE = slope_east_lab[run_ind,:]
sN = slope_north_lab[run_ind,:]
sE = np.where(np.isfinite(sE), sE, 0.0)
sN = np.where(np.isfinite(sN), sN, 0.0)
elev_m_lab = slope_to_elev_wavelet(sE,sN,water_depth_m,sampling_rate_PSS,aperture_mtf_curve=mtf_curve,fmin_Hz=f_hp)

sE = slope_east_emp[run_ind,:]
sN = slope_north_emp[run_ind,:]
sE = np.where(np.isfinite(sE), sE, 0.0)
sN = np.where(np.isfinite(sN), sN, 0.0)
elev_m_emp = slope_to_elev_wavelet(sE,sN,water_depth_m,sampling_rate_PSS,aperture_mtf_curve=mtf_curve,fmin_Hz=f_hp)

slope_lim_val = 0.3

t_start = 456   # start of the displayed 20-s window (an energetic stretch of
                # this run; the lidar timeseries is not collocated so it is not
                # overlaid -- spectral comparison is in elevation_omnispect.pdf)

t = t - t_start

fig, ax1 = plt.subplots(figsize=(fullwidth, fullwidth/2))

plt.xlim(0,20)

# Slope timeseries (demeaned: the raw series carry a fixed viewing-angle tilt
# of ~0.1 rad on s_E and ~0.56 rad on s_N, which would push s_N off the plot
# and is removed inside slope_to_elev_wavelet before the inversion anyway).
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

# elevation timeseries: E-PSS wavelet long-wave inversion (identified by the
# red right-hand axis -- no legend entry needed).
ax2.plot(t, elev_m_emp, linewidth=2, color=color_list[2])
ax2.set_ylabel('$\eta$ [m]',color=color_list[2])
ax2.tick_params(axis='y',labelcolor=color_list[2])

ax2.set_ylim(-2*slope_lim_val,2*slope_lim_val)

plt.savefig('../_figures/wave_slope_elev_timeseries.pdf',bbox_inches='tight')

# %%

# E-PSS omnidirectional spectrum is the directionally-complete (S_sx+S_sy)/k^2
# estimator (omni_complete_spectrum), one curve per gain treatment. It carries
# the full directional spread (no single-direction projection loss), so it is
# THE E-PSS omni product -- not a separate overlay. Same aperture-MTF gain^2 and
# f_hp high-pass as the dataset (compute_all_omnidirectional_spectra.py). The
# elevation timeseries panel above still uses the wavelet inversion (this
# estimator is a frequency spectrum only).
f_Hz_lidar, Pxx_den_lidar = signal.welch(elev_m_lidar[0,:,run_ind], sampling_rate_lidar, nperseg=nperseg)
f_Hz, Pxx_den_no, _ = omni_complete_spectrum(slope_east_no[run_ind], slope_north_no[run_ind],
    water_depth_m, sampling_rate_PSS, aperture_mtf_curve=mtf_curve, highpass_peak_fraction=0.5, nperseg=nperseg)
_, Pxx_den_lab, _ = omni_complete_spectrum(slope_east_lab[run_ind], slope_north_lab[run_ind],
    water_depth_m, sampling_rate_PSS, aperture_mtf_curve=mtf_curve, highpass_peak_fraction=0.5, nperseg=nperseg)
_, Pxx_den_emp, _ = omni_complete_spectrum(slope_east_emp[run_ind], slope_north_emp[run_ind],
    water_depth_m, sampling_rate_PSS, aperture_mtf_curve=mtf_curve, highpass_peak_fraction=0.5, nperseg=nperseg)

# Start each curve at the first bin with f > 0.05 Hz (lidar and E-PSS are on
# different Welch grids, so the start index is computed per grid).
i_lid = int(np.argmax(f_Hz_lidar > 0.05))
i_pss = int(np.argmax(f_Hz > 0.05))

fig = plt.figure(figsize=(fullwidth/2,fullwidth/2*4/3))

plt.plot(f_Hz_lidar[i_lid:],Pxx_den_lidar[i_lid:],color='black',linewidth=2,label="lidar")
plt.plot(f_Hz[i_pss:],Pxx_den_no[i_pss:],color=color_list[0],linewidth=2,alpha=0.75,label="E-PSS, no gain")
plt.plot(f_Hz[i_pss:],Pxx_den_lab[i_pss:],color=color_list[1],linewidth=2,alpha=0.75,label="E-PSS, lab gain")
plt.plot(f_Hz[i_pss:],Pxx_den_emp[i_pss:],color=color_list[2],linewidth=2,alpha=0.75,label="E-PSS, emp. gain")

plt.grid(which='major', linestyle='-', linewidth=0.75)
plt.grid(which='minor', linestyle=':', linewidth=0.75)

plt.xlim(1e-2,1e1)
plt.ylim(1e-4,1e0)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('f [Hz]')
plt.ylabel(r'$F_{\eta\eta}(f)$ [m$^2$Hz$^{-1}$]')

plt.legend(loc='lower left')

plt.savefig('../_figures/elevation_omnispect.pdf',bbox_inches='tight')

print(f'[run {run_ind}] E-PSS long-wave eta: std={np.std(elev_m_emp):.3f} m')

