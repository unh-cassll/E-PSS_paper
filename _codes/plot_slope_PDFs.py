"""
Created on Mon Sep 15 15:40:31 2025

@author: nathanlaxague
"""

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import netCDF4 as nc

from subroutines.utils import *

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

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

slope_centers = ds_no["slope_centers"][:]*-1
slope_histogram_crosswind_upwind_no = ds_no["slope_histogram_crosswind_upwind"][:]
slope_histogram_crosswind_upwind_lab = ds_lab["slope_histogram_crosswind_upwind"][:]
slope_histogram_crosswind_upwind_emp = ds_emp["slope_histogram_crosswind_upwind"][:]

panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)']

slope_PDF = np.nan*np.ones((190,200,3,2))

slope_PDF[:,:,0,0] = np.trapezoid(slope_histogram_crosswind_upwind_no,x=slope_centers,axis=2)
slope_PDF[:,:,1,0] = np.trapezoid(slope_histogram_crosswind_upwind_lab,x=slope_centers,axis=2)
slope_PDF[:,:,2,0] = np.trapezoid(slope_histogram_crosswind_upwind_emp,x=slope_centers,axis=2)

slope_PDF[:,:,0,1] = np.trapezoid(slope_histogram_crosswind_upwind_no,x=slope_centers,axis=1)
slope_PDF[:,:,1,1] = np.trapezoid(slope_histogram_crosswind_upwind_lab,x=slope_centers,axis=1)
slope_PDF[:,:,2,1] = np.trapezoid(slope_histogram_crosswind_upwind_emp,x=slope_centers,axis=1)

slope_PDF = np.abs(slope_PDF)

U_centers = np.float64(np.arange(2,13,4))
dU = np.float64(2.0)

U_low_string = (U_centers-dU/2).astype(str)
U_high_string = (U_centers+dU/2).astype(str)

U_centers_string = U_centers.astype(str)
dU_string = dU.astype(str)

slope_PDF_binned = np.nan*np.ones((len(U_centers),200,3,2))

dolp_gain_choices = ['no gain','lab gain','empirical gain']

for i in np.arange(len(U_centers)):
    
    inds = (U10_m_s > U_centers[i] - dU/2) & (U10_m_s <= U_centers[i] + dU/2)
    slope_PDF_binned[i,:,:,:] = np.mean(slope_PDF[inds,:,:,:],axis=0)

counter = 0

fig, axs = plt.subplots(len(U_centers), 2, sharex=True, sharey=True, figsize=(12, 13))

for j in np.arange(len(U_centers)):
    
    wave_slope_PDF, mss_cross, mss_up = compute_gram_charlier_slope_pdf(U_centers[j])
    
    for k in np.arange(3):
        
        axs[j,0].plot(slope_centers,slope_PDF_binned[j,:,k,0],color=color_list[k],label=dolp_gain_choices[k],linewidth=2)
        axs[j,1].plot(slope_centers,slope_PDF_binned[j,:,k,1],color=color_list[k],label=dolp_gain_choices[k],linewidth=2)
    
    wind_string = U_low_string[j] + ' < U ≤ ' + U_high_string[j]
    axs[j, 1].text(-0.47,33, wind_string,
             fontsize=12,
             color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=1))
    
    axs[j, 0].plot(wave_slope_PDF['slope_up'],wave_slope_PDF.integrate('slope_cross'),'--',linewidth=2,color='black',label=r'Bréon & Henriot [2006]')
    axs[j, 0].set_ylabel('P(slope)')
    
    axs[j, 1].plot(wave_slope_PDF['slope_cross'],wave_slope_PDF.integrate('slope_up'),'--',linewidth=2,color='black')

    for l in np.arange(2):
        
        axs[j,l].text(0.95,0.95,panel_labels[counter],fontsize=12,ha='center',va='center',transform=axs[j,l].transAxes)
        counter = counter + 1
        
    plt.xlim(-0.5,0.5)
    plt.ylim(1e-1,5e1)
    
    plt.yscale('log')
    
axs[len(U_centers)-1,0].set_xlabel('upwind slope')
axs[len(U_centers)-1,1].set_xlabel('crosswind slope')

axs[0,0].legend(loc='upper left')

plt.tight_layout()

plt.savefig('../_figures/slope_distributions_binned_by_wind.pdf',bbox_inches='tight')


