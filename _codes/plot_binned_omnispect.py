"""
Created on Tue Sep 16 19:45:25 2025

@author: nathanlaxague
"""

import numpy as np
import xarray as xr

import netCDF4 as nc

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from subroutines.utils import *

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

# Set custom property cycle colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611'])

panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)']

path = '../_data/'

ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')

ds_omnispect = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')

U10_m_s = ds_other["COARE_U10"]

f_Hz_slope = ds_emp['f_Hz'][:]
f_Hz_slope = f_Hz_slope.data
k_rad_m_slope = ds_emp['k_rad_m'][:]
theta_rad = ds_emp['theta_rad'][:]
S_f_theta = ds_emp['S_f_theta'][:]
S_k_theta = ds_emp['S_k_theta'][:]

S_k_theta[:,:,len(k_rad_m_slope)-1] = np.nan

df = np.median(np.diff(f_Hz_slope))
dk = np.median(np.diff(k_rad_m_slope))
dtheta = np.median(np.diff(theta_rad))

S_f = np.sum(S_f_theta,axis=1)*dtheta
S_k = np.sum(np.reshape(k_rad_m_slope,(1,1,len(k_rad_m_slope)))*S_k_theta,axis=1)*dtheta
B_k = np.sum(np.reshape(k_rad_m_slope,(1,1,len(k_rad_m_slope)))**2*S_k_theta,axis=1)*dtheta

f_Hz = ds_omnispect['frequency'][:].data
F_f_m2_Hz_empirical_gain = ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data
   
f_inds = f_Hz < 0.05
F_f_m2_Hz_empirical_gain[[0,1],:] = np.nan
F_f_m2_Hz_empirical_gain[f_inds,:] = np.nan
df = np.median(np.diff(f_Hz))    

U_centers = np.float64(np.arange(2,14,2))
dU = np.float64(2.0)
U_boundaries = np.arange(U_centers[0]-dU/2,U_centers[len(U_centers)-1]+dU/2+dU,dU)

U_low_string = (U_centers-dU/2).astype(str)
U_high_string = (U_centers+dU/2).astype(str)

U_centers_string = U_centers.astype(str)
dU_string = dU.astype(str)

F_f_binned = np.nan*np.ones((len(U_centers),len(f_Hz)))
S_f_binned = np.nan*np.ones((len(U_centers),len(f_Hz_slope)))
S_k_binned = np.nan*np.ones((len(U_centers),len(k_rad_m_slope)))

dolp_gain_choices = ['no gain','lab gain','empirical gain']

for i in np.arange(len(U_centers)):
    
    inds = (U10_m_s > U_centers[i] - dU/2) & (U10_m_s <= U_centers[i] + dU/2)
    F_f_binned[i,:] = np.nanmean(F_f_m2_Hz_empirical_gain[:,inds],axis=1).T
    S_f_binned[i,:] = np.mean(S_f[inds,:],axis=0)
    S_k_binned[i,:] = np.mean(S_k[inds,:],axis=0)
    
T_s = f_Hz**-1
T_s_slope = f_Hz_slope**-1
h_m = 15

C_m_s_disp, Cg_m_s_disp = lindisp_with_current(2*np.pi*f_Hz,h_m,0)
k_rad_m_disp = 2*np.pi*f_Hz / C_m_s_disp

C_m_s_slope_disp, Cg_m_s_slope_disp = lindisp_with_current(2*np.pi*f_Hz_slope,h_m,0)
k_rad_m_slope_disp = 2*np.pi*f_Hz_slope / C_m_s_slope_disp

k_vec = np.reshape(k_rad_m_slope,(1,len(k_rad_m_slope)))
k_slope_disp_vec = np.reshape(k_rad_m_slope_disp,(1,len(k_rad_m_slope_disp)))
k_disp_vec = np.reshape(k_rad_m_disp,(1,len(k_rad_m_disp)))

F_k_disp_binned = np.reshape(Cg_m_s_disp/(np.pi),(1,len(k_rad_m_disp)))*F_f_binned
B_k_disp_binned = np.reshape(k_rad_m_disp,(1,len(k_rad_m_disp)))**3*F_k_disp_binned

F_f_slope_disp_binned = np.reshape(k_rad_m_slope_disp,(1,len(k_rad_m_slope_disp)))**-2*S_f_binned
B_f_slope_disp_binned = np.reshape(f_Hz_slope,(1,len(k_rad_m_slope_disp)))*S_f_binned
B_f_binned = np.reshape(k_rad_m_disp,(1,len(k_rad_m_disp)))**2*np.reshape(f_Hz,(1,len(f_Hz)))*F_f_binned

cmap = plt.get_cmap('cividis')
color_min = np.min(U_centers)
color_max = np.max(U_centers)    
colors = [cmap(j) for j in np.linspace(0,1, len(U_centers))]  # Get colors from the colormap

f_lims = [1e-2,2e1]
Ff_lims = [1e-10,1e1]
Bf_lims = [5e-6,5e-2]

f_cut = 0.4
f_inds = f_Hz < f_cut
k_inds = k_rad_m_disp < (2*np.pi*f_cut)**2/9.81

f_slope_inds = f_Hz_slope > 0.5

k_lims = [1e-2,1e3]
Fk_lims = [1e-12,1e1]
Bk_lims = [5e-6,5e-2]

f_eq_lims = [2e-1,7e-1]
f_sat_lims = [4e-1,1.5e0]

k_eq_lims = [2e-1,1.5e0]
k_sat_lims = [5e-1,1.5e1]

lw_thick = 3.5
lw_thin = 2.5

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i in np.arange(len(U_centers)):
    
    # Frequency elevation spectra
    axs[0,0].loglog(f_Hz[f_inds], F_f_binned[i,f_inds].T, color='black',linewidth=lw_thick)
    axs[0,0].loglog(f_Hz_slope[f_slope_inds], F_f_slope_disp_binned[i,f_slope_inds].T, color='black',linewidth=lw_thick)
    
    # Frequency saturation spectra
    axs[1,0].loglog(f_Hz[f_inds], B_f_binned[i,f_inds].T, color='black',linewidth=lw_thick)
    axs[1,0].loglog(f_Hz_slope[f_slope_inds], B_f_slope_disp_binned[i,f_slope_inds].T, color='black',linewidth=lw_thick)
    
    # Wavenumber elevation spectra
    axs[0,1].loglog(k_rad_m_disp[k_inds],(F_k_disp_binned[i,k_inds]).T, color='black',linewidth=lw_thick)
    axs[0,1].loglog(k_rad_m_slope,(k_vec**-2*S_k_binned[i,:]).T, color='black',linewidth=lw_thick)
    
    # Wavenumber saturation spectra
    axs[1,1].loglog(k_rad_m_disp[k_inds],(B_k_disp_binned[i,k_inds]).T, color='black',linewidth=lw_thick)
    axs[1,1].loglog(k_rad_m_slope,(k_vec*S_k_binned[i,:]).T, color='black',linewidth=lw_thick)

for i in np.arange(len(U_centers)):
    
    # Frequency elevation spectra
    axs[0,0].loglog(f_Hz[f_inds], F_f_binned[i,f_inds].T, color=colors[i],linewidth=lw_thin)
    axs[0,0].loglog(f_Hz_slope[f_slope_inds], F_f_slope_disp_binned[i,f_slope_inds].T, color=colors[i],linewidth=lw_thin)
    axs[0,0].loglog(f_eq_lims,1e-2*np.power(f_eq_lims,-4),'--',color='black')
    axs[0,0].loglog(f_sat_lims,5e-3*np.power(f_sat_lims,-5),':',color='black')

    axs[0,0].set_xlim(f_lims)
    axs[0,0].set_ylim(Ff_lims)
    axs[0,0].set_ylabel(r'F(f) [m$^2$Hz$^{-1}$]')
    axs[0,0].set_xticklabels([])
    
    # Frequency saturation spectra
    axs[1,0].loglog(f_Hz[f_inds], B_f_binned[i,f_inds].T, color=colors[i],linewidth=lw_thin)
    axs[1,0].loglog(f_Hz_slope[f_slope_inds], B_f_slope_disp_binned[i,f_slope_inds].T, color=colors[i],linewidth=lw_thin)
    axs[1,0].loglog(f_eq_lims,4e-2*np.power(f_eq_lims,1),'--',color='black')
    axs[1,0].loglog(f_sat_lims,2e-2*np.power(f_sat_lims,0),':',color='black')

    axs[1,0].set_xlim(f_lims)
    axs[1,0].set_ylim(Bf_lims)
    axs[1,0].set_ylabel(r'(2$\pi$f)$^5$g$^{-2}$F(f) [rad]')
    
    # Wavenumber elevation spectra
    axs[0,1].loglog(k_rad_m_disp[k_inds],(F_k_disp_binned[i,k_inds]).T, color=colors[i],linewidth=lw_thin)
    axs[0,1].loglog(k_rad_m_slope,(k_vec**-2*S_k_binned[i,:]).T, color=colors[i],linewidth=lw_thin)
    axs[0,1].loglog(k_eq_lims,5e-2*np.power(k_eq_lims,-2.5),'--',color='black')
    axs[0,1].loglog(k_sat_lims,5e-2*np.power(k_sat_lims,-3),':',color='black')

    axs[0,1].set_xlim(k_lims)
    axs[0,1].set_ylim(Fk_lims)
    axs[0,1].set_ylabel(r'F(k) [m$^3$]')
    axs[0,1].set_xticklabels([])
    
    # Wavenumber saturation spectra
    axs[1,1].loglog(k_rad_m_disp[k_inds],(B_k_disp_binned[i,k_inds]).T, color=colors[i],linewidth=lw_thin)
    axs[1,1].loglog(k_rad_m_slope,(k_vec*S_k_binned[i,:]).T, color=colors[i],linewidth=lw_thin)
    axs[1,1].loglog(k_eq_lims,2e-2*np.power(k_eq_lims,0.5),'--',color='black')
    axs[1,1].loglog(k_sat_lims,2e-2*np.power(k_sat_lims,0),':',color='black')

    axs[1,1].set_xlim(k_lims)
    axs[1,1].set_ylim(Bk_lims)
    axs[1,1].set_ylabel(r'k$^3$F(k) [rad]')
    

axs[1,0].set_xlabel('f [Hz]')
axs[1,1].set_xlabel(r'k [rad m$^{-1}$]')

counter = 0
    
for i in np.arange(2):
    for j in np.arange(2):
        axs[i,j].grid(which='major', linestyle='-', linewidth=0.75)  # Major gridlines with solid linestyle
        axs[i,j].grid(which='minor', linestyle=':', linewidth=0.75)  # Minor gridlines with dotted linestyle
        
        axs[i,j].text(0.05,0.95,panel_labels[counter],fontsize=12,ha='center',va='center',transform=axs[i,j].transAxes)
        counter = counter + 1

plt.tight_layout()

norm = BoundaryNorm(U_boundaries, cmap.N)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)    
cbar = fig.colorbar(sm, ax=axs, location='top', orientation='horizontal', fraction=0.05, pad=0.03,aspect=50)
cbar.set_label(r'U$_{10}$ [m s$^{-1}$]')

plt.savefig('../_figures/omnidirectional_spectra_binned_by_wind.pdf',bbox_inches='tight')
