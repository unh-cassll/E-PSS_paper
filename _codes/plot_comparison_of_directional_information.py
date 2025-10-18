"""
Created on Wed Sep 17 07:58:12 2025

@author: nathanlaxague
"""

import sys
sys.path.append('subroutines/')

import numpy as np
import xarray as xr

import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm

import seaborn as sns

from scipy import stats
import scipy.signal as signal

import scientimate

from utils import *

g = 9.81;

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

panel_labels = ['(a)','(b)']

path = '../_data/'

fn = path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc'
ds = nc.Dataset(fn)

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
    
ds_EPSS_spect = xr.open_dataset(path+'ASIT2019_EPSS_directional_spectra.nc')

elev_m = ds['elev_m'][:]

f_Hz_Pyxis = ds['f_Hz'][:]
theta_rad_Pyxis = ds['theta_rad'][:]
S_f_theta_Pyxis = ds['S_f_theta'][:]

f_Hz_ADCP = ds_other['f_Hz_ADCP'][:]
theta_rad_ADCP = ds_other['theta_rad'][:]
Fftheta_m2_Hz_rad_ADCP = ds_other['F_f_theta_m2_Hz_rad_ADCP'][:]

U10_m_s = ds_other["COARE_U10"][:]

theta_rad_ADCP = theta_rad_ADCP[np.arange(0,len(theta_rad_ADCP)-1)]

bigtheta = np.concatenate((theta_rad_ADCP-2*np.pi,theta_rad_ADCP,theta_rad_ADCP+2*np.pi),axis=0)*180/np.pi

inds_keep = (bigtheta >= -180) & (bigtheta <= 180)
theta_deg_ADCP = bigtheta[inds_keep]

f_Hz = ds_EPSS_spect['frequency'].data

num_runs = 190
num_f = len(f_Hz)
fs_Hz = 2*f_Hz[len(f_Hz)-1]
nperseg = np.int16((num_f-1)*2)

MWD_ADCP = np.nan*np.ones(num_runs)
MWD_EPSS = MWD_ADCP.copy()
MWD_EPSS_short = MWD_ADCP.copy()
MWD_short = MWD_ADCP.copy()
Tm01_EPSS = MWD_ADCP.copy()
Tm01_ADCP = MWD_ADCP.copy()
ind_peak_ADCP = np.int16(np.ones(num_runs))
ind_peak_EPSS = ind_peak_ADCP.copy()
SPREAD_ADCP = np.nan*np.ones((num_runs,len(f_Hz_ADCP)))
SPREAD_EPSS = np.nan*np.ones((num_runs,num_f))

Ff_EPSS = np.nan*np.ones((num_runs,num_f))

f_low_filt = 0.05
f_high_filt = 1

smoothnum = 5

theta_halfwidth = 120

f_cut_high = 0.45

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
    
    MWD_short[run_ind] = short_mwd
    
    Fftheta_m2_Hz_rad_ADCP_particular = Fftheta_m2_Hz_rad_ADCP[:,:,run_ind].T
    
    Fftheta_m2_Hz_rad_ADCP_particular = Fftheta_m2_Hz_rad_ADCP_particular[:,np.arange(0,len(theta_deg_ADCP)-1)]
    
    bigFftheta = np.concatenate((Fftheta_m2_Hz_rad_ADCP_particular,Fftheta_m2_Hz_rad_ADCP_particular,Fftheta_m2_Hz_rad_ADCP_particular),axis=1)
        
    Fftheta_m2_Hz_rad_ADCP_particular = bigFftheta[:,inds_keep]
    
    # creating dataset (ADCP spectrum)
    dataset_ADCP = xr.Dataset(
        coords = {"frequency": f_Hz_ADCP, "direction": theta_deg_ADCP},
        data_vars = {
            "F_ADCP": (["frequency", "direction"], Fftheta_m2_Hz_rad_ADCP_particular*np.pi/180)
        }
    )
    
    F_ADCP = dataset_ADCP.F_ADCP
    Ff_ADCP = F_ADCP.integrate('direction')
    
    F_EPSS = ds_EPSS_spect['F_f_d'][:,:,run_ind]
    
    Ff_EPSS[run_ind,:] = F_EPSS.integrate('direction')
    
    inds_exclude = (F_ADCP["frequency"].data > f_cut_high) | (F_ADCP["frequency"].data < f_low_filt)
    F_ADCP.data[inds_exclude,:] = 0

    inds_exclude = (F_EPSS["frequency"].data > f_cut_high) | (F_EPSS["frequency"].data < f_low_filt)
    F_EPSS.data[inds_exclude,:] = 0
               
    mwd_EPSS, spread_EPSS = compute_mean_wave_direction_and_spreading(F_EPSS,theta_halfwidth,smoothnum)
    MWD_EPSS[run_ind] = mwd_EPSS
    SPREAD_EPSS[run_ind,:] = spread_EPSS
    
    total_energy = F_ADCP.integrate('frequency').integrate('direction')
    
    if total_energy > 0:
        mwd_ADCP, spread_ADCP = compute_mean_wave_direction_and_spreading(F_ADCP,theta_halfwidth,smoothnum=3)
        
        Ff_ADCP = F_ADCP.integrate('direction').data
        f_E = np.sum(Ff_ADCP)/np.sum(f_Hz_ADCP**-1*Ff_ADCP)
        f_diff = np.abs(f_E-f_Hz_ADCP)
        ind = np.argmin(f_diff)
        
        ind_peak_ADCP[run_ind] = ind
        MWD_ADCP[run_ind] = mwd_ADCP
        SPREAD_ADCP[run_ind,:] = spread_ADCP
        
    else:
        ind_peak_ADCP[run_ind] = 0
        MWD_ADCP[run_ind] = np.nan
        SPREAD_ADCP[run_ind,:] = np.nan*np.ones((1,43))
    
    f_Hz_EPSS, Ff_m2_Hz_EPSS = signal.welch(elev_m[run_ind,:], fs_Hz, nperseg=nperseg)
    f_Hz_EPSS[0] = np.nan
    f_E = np.nansum(Ff_m2_Hz_EPSS)/np.nansum(f_Hz_EPSS**-1*Ff_m2_Hz_EPSS)
    f_diff = np.abs(f_E-f_Hz_EPSS)
    f_diff[0] = 1e3
    ind = np.argmin(f_diff)
    ind_peak_EPSS[run_ind] = ind
    
    Ff_EPSS_val = F_EPSS.integrate('direction')
    Ff_EPSS_val = Ff_EPSS_val.data
    
    Tm01_ADCP[run_ind] = F_ADCP.integrate('direction').integrate('frequency')/np.trapezoid(f_Hz_ADCP*Ff_ADCP,x=f_Hz_ADCP)
    Tm01_EPSS[run_ind] = F_EPSS.integrate('direction').integrate('frequency')/np.trapezoid(F_EPSS['frequency'][:]*Ff_EPSS_val,x=F_EPSS['frequency'][:])

h_m_ADCP = 18.3
h_m_EPSS = 15

k_rad_m_disp_ADCP, _, C_m_s_disp_ADCP, Cg_m_s_disp_ADCP = scientimate.wavedispersionds(h_m_ADCP, Tm01_ADCP, Uc=0)
k_rad_m_disp_EPSS, _, C_m_s_disp_EPSS, Cg_m_s_disp_EPSS = scientimate.wavedispersionds(h_m_EPSS, Tm01_EPSS, Uc=0)

# Account for wave refraction (coastline is approximately East-West, so MWD is already shore-relative)
MWD_ADCP_shifted = np.asin(C_m_s_disp_EPSS/C_m_s_disp_ADCP*np.sin(MWD_ADCP*np.pi/180))*180/np.pi

# Unwrap angular differences arising from 180 degree ambiguity
inds_northerly = MWD_EPSS < -90
MWD_EPSS[inds_northerly] = MWD_EPSS[inds_northerly] + 180
inds_northerly = MWD_EPSS > 90
MWD_EPSS[inds_northerly] = MWD_EPSS[inds_northerly] - 180
MWD_diff = MWD_EPSS-MWD_ADCP_shifted
# MWD_diff[MWD_diff>90] = MWD_diff[MWD_diff>90] - 180
# MWD_diff[MWD_diff<-90] = MWD_diff[MWD_diff<-90] + 180

# %%

U10_bin_edges = np.arange(1,15,2)
U10_bin_centers = U10_bin_edges[0:len(U10_bin_edges)-1] + np.diff(U10_bin_edges)/2

inds = ~np.isnan(MWD_diff)

bin_means, bin_edges, binnumber = stats.binned_statistic(U10_m_s[inds],MWD_diff[inds], statistic='mean', bins=U10_bin_edges)
bin_std, _, _ = stats.binned_statistic(U10_m_s[inds],MWD_diff[inds], statistic='std', bins=U10_bin_edges)
bin_counts, _, _ = stats.binned_statistic(U10_m_s[inds], MWD_diff[inds], statistic='count', bins=U10_bin_edges)

bin_95CI = 1.96*bin_std/bin_counts
bin_upper = bin_means + bin_95CI
bin_lower = bin_means - bin_95CI

fig = plt.figure(figsize=(6,6))
plt.fill_between(U10_bin_centers, bin_upper, bin_lower, color=color_list[2], alpha=0.25)
plt.plot(U10_bin_centers,bin_means,'-',color=color_list[2],linewidth=1,label=r'$\theta_{E-PSS}-\theta_{ADCP}$')
plt.plot([0,16],[0,0],'--',color='gray')
plt.xlim(0,14)
plt.yticks(np.arange(-360,360,30))
plt.ylim(-90,90)
plt.xlabel(r'$U_{10}$ [m s$^{-1}$]')
plt.ylabel(r'$\Delta\theta_0$ [$\circ$]')
plt.legend()

plt.savefig('../_figures/delta_theta_nought.pdf',bbox_inches='tight')


# %%

SPREAD_ADCP_peak = np.nan*np.ones(num_runs)
SPREAD_EPSS_peak = SPREAD_ADCP_peak.copy()
f_E = SPREAD_ADCP_peak.copy()

for run_num in np.arange(0,num_runs):
    SPREAD_ADCP_peak[run_num] = SPREAD_ADCP[run_num,ind_peak_ADCP[run_num]]
    SPREAD_EPSS_peak[run_num] = SPREAD_EPSS[run_num,ind_peak_EPSS[run_num]]
    f_E[run_num] = f_Hz_EPSS[ind_peak_EPSS[run_num]]

SPREAD_peak = np.nan*np.ones((num_runs,2))
SPREAD_peak[:,0] = SPREAD_ADCP_peak
SPREAD_peak[:,1] = SPREAD_EPSS_peak

labels = ['ADCP','E-PSS']

run_ind = 135
spread_ADCP = SPREAD_ADCP[run_ind,:]
spread_EPSS = SPREAD_EPSS[run_ind,:]

fig,axs = plt.subplots(1,2,figsize=(12,5))
axs[0].plot(F_ADCP["frequency"],spread_ADCP,label="ADCP",linewidth=2)
axs[0].plot(F_EPSS["frequency"],spread_EPSS,label="E-PSS",linewidth=2)
axs[0].plot(f_Hz_EPSS[ind_peak_EPSS[run_ind]]*np.float64([1.0,1.0]),[0,90])
axs[0].set_xscale('log')
axs[0].set_yticks(np.arange(0,360,15))
axs[0].set_ylim(0,90)
axs[0].set_xlim(1e-2,1e0)
axs[0].set_xlabel('f [Hz]')
axs[0].set_ylabel(r'$\sigma_{\theta}$ [$\circ$]')
axs[0].text(f_Hz_EPSS[ind_peak_EPSS[run_ind]]*0.8,82.5,r'$f_E$',color=color_list[2])
axs[0].legend()

axs[0].grid(which='major', linestyle='-', linewidth=0.75)  # Major gridlines with solid linestyle
axs[0].grid(which='minor', linestyle=':', linewidth=0.75)  # Minor gridlines with dotted linestyle

for n in np.arange(2):
    
    values = SPREAD_peak[:,n]
    inds = ~np.isnan(values)

    bin_means, bin_edges, binnumber = stats.binned_statistic(U10_m_s[inds],values[inds], statistic='mean', bins=U10_bin_edges)
    bin_std, _, _ = stats.binned_statistic(U10_m_s[inds],values[inds], statistic='std', bins=U10_bin_edges)
    bin_counts, _, _ = stats.binned_statistic(U10_m_s[inds], values[inds], statistic='count', bins=U10_bin_edges)

    bin_95CI = 1.96*bin_std/bin_counts
    bin_upper = bin_means + bin_95CI
    bin_lower = bin_means - bin_95CI

    axs[1].fill_between(U10_bin_centers, bin_upper, bin_lower, color=color_list[n], alpha=0.25)
    axs[1].plot(U10_bin_centers,bin_means,'-',linewidth=2,label=labels[n])


axs[1].set_yticks(np.arange(0,360,15))
axs[1].set_ylim(0,90)
axs[1].set_xlim(0,14)
axs[1].set_xlabel(r'$U_{10}$ [m s$^{-1}$]')
axs[1].set_ylabel(r'$\sigma_{\theta}$, evaluated at $f=f_E$ [$\circ$]')

for n in np.arange(2):
    
    axs[n].text(0.05,0.95,panel_labels[n],fontsize=12,ha='center',va='center',transform=axs[n].transAxes)
    
plt.savefig('../_figures/directional_spreading_comparison.pdf',bbox_inches='tight')

