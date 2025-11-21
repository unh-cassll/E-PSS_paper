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


import seaborn as sns


from make_plots_v2 import plot_directional_spectrum


from utils import *


g = 9.81;


sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")


color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


path = '../_data/'


fn = path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc'
ds = nc.Dataset(fn)


ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')


ds_EPSS_spect = xr.open_dataset(path+'ASIT2019_EPSS_directional_spectra.nc')
    
elev_m = ds['elev_m'][:]
slope_north = ds['slope_north'][:]
slope_east = ds['slope_east'][:]


f_Hz = ds['f_Hz'][:]


fs_Hz = np.floor(2*f_Hz[len(f_Hz)-1])


f_Hz_Pyxis = ds['f_Hz'][:]
k_rad_m_Pyxis = ds['k_rad_m'][:]
theta_rad_Pyxis = ds['theta_rad'][:]
S_f_theta_Pyxis = ds['S_f_theta'][:]
S_k_theta_Pyxis = ds['S_k_theta'][:]


f_Hz_ADCP = ds_other['f_Hz_ADCP'][:]
theta_rad_ADCP = ds_other['theta_rad'][:]
Fftheta_m2_Hz_rad_ADCP = ds_other['F_f_theta_m2_Hz_rad_ADCP'][:]


U10 = ds_other["COARE_U10"]
winddir = ds_other["COARE_Wdir"]


U_sfc_mag_m_s = ds_other["U_sfc_mag_m_s"]
U_sfc_dir_deg = ds_other["U_sfc_dir_deg"]


num_samples = np.size(elev_m,axis=1)


nfft = num_samples/4
nperseg = nfft/2


run_ind = 165


lowcut_filter = 0.05
highcut_filter = 1


f_cut_high = 0.35
f_cut_low = 0.05


theta_halfwidth = 120
smoothnum = 3


S_f_theta_Pyxis_particular = S_f_theta_Pyxis[run_ind,:,:].T
S_k_theta_Pyxis_particular = S_k_theta_Pyxis[run_ind,:,:].T


Fftheta_m2_Hz_rad_ADCP_particular = Fftheta_m2_Hz_rad_ADCP[:,:,run_ind].T


Fftheta_m2_Hz_rad_ADCP_particular = Fftheta_m2_Hz_rad_ADCP_particular[:,np.arange(0,len(theta_rad_ADCP)-1)]
theta_rad_ADCP = theta_rad_ADCP[np.arange(0,len(theta_rad_ADCP)-1)]


bigtheta = np.concatenate((theta_rad_ADCP-2*np.pi,theta_rad_ADCP,theta_rad_ADCP+2*np.pi),axis=0)*180/np.pi
bigFftheta = np.concatenate((Fftheta_m2_Hz_rad_ADCP_particular,Fftheta_m2_Hz_rad_ADCP_particular,Fftheta_m2_Hz_rad_ADCP_particular),axis=1)


inds_keep = (bigtheta >= -180) & (bigtheta <= 180)
theta_deg_ADCP = bigtheta[inds_keep]
Fftheta_m2_Hz_rad_ADCP = bigFftheta[:,inds_keep]


# creating dataset (ADCP spectrum)
dataset_ADCP = xr.Dataset(
    coords = {"frequency": f_Hz_ADCP, "direction": theta_deg_ADCP},
    data_vars = {
        "F_ADCP": (["frequency", "direction"], Fftheta_m2_Hz_rad_ADCP*np.pi/180)
    }
)


k_disp = (2*np.pi*f_Hz_Pyxis)**2/g
k_disp_mat = np.tile(k_disp,(len(theta_rad_Pyxis),1)).T
Fftheta_m2_Hz_rad_Pyxis = S_f_theta_Pyxis_particular*k_disp_mat**-2


Fftheta_m2_Hz_rad_Pyxis_shifted = np.concatenate((Fftheta_m2_Hz_rad_Pyxis[:,np.arange(36,72)],Fftheta_m2_Hz_rad_Pyxis[:,np.arange(0,36)]),axis=1)
theta_rad_Pyxis_shifted = np.concatenate((theta_rad_Pyxis[np.arange(36,72)]-2*np.pi,theta_rad_Pyxis[np.arange(0,36)]))


# creating dataset (Pyxis frequency spectrum)
dataset_Pyxis_frequency = xr.Dataset(
    coords = {"frequency": f_Hz_Pyxis, "direction": 180/np.pi*theta_rad_Pyxis_shifted},
    data_vars = {
        "Ffd": (["frequency", "direction"], Fftheta_m2_Hz_rad_Pyxis_shifted*np.pi/180)
    }
)


Ffd = dataset_Pyxis_frequency.Ffd


F_EPSS = ds_EPSS_spect['F_f_d'][:,:,run_ind]


Ff_ADCP = dataset_ADCP.F_ADCP.integrate('direction')
F_ADCP = dataset_ADCP.F_ADCP


inds_exclude = (F_ADCP["frequency"].data > f_cut_high) | (F_ADCP["frequency"].data < f_cut_low)
F_ADCP.data[inds_exclude,:] = 0


inds_exclude = (F_EPSS["frequency"].data > f_cut_high) | (F_EPSS["frequency"].data < f_cut_low)
F_EPSS.data[inds_exclude,:] = 0


D_ADCP = ((F_ADCP.T / F_ADCP.integrate("direction")).rolling(frequency=smoothnum, center=True).median()).T


Df_Pyxis = ((dataset_Pyxis_frequency.Ffd.T / dataset_Pyxis_frequency.Ffd.integrate("direction")).rolling(frequency=9, center=True).median()).T


D_EPSS = ((F_EPSS.T / F_EPSS.integrate("direction")).rolling(frequency=smoothnum, center=True).median()).T


# %%
# Directional wave spectrum
# -------------------------
#
# Now, we show the corresponding directional wave spectra
# :math:`E(f,\theta)` for each method. Recalling that this quantity is
# constructed as:
#
# .. math:: E(f,\theta) = S(f) D(f,\theta)
#
# For the Fourier-based methods, the frequency spectrum :math:`S(f)` is
# simply the Fourier transform of the surface elevation signal, i.e.,
# :math:`E_{zz}(f)`, the auto-spectrum of :math:`z(t)`. For the wavelet
# method, :math:`S(f)` is obtained time-averaging the squared
# wavelet amplitudes. See `Directional distribution and directional spectrum <https://extended-wdm.readthedocs.io/en/latest/maths.html#directional-distribution-and-directional-spectrum>`_.
#


vmin, vmax = -3.5,-1.5
axes_kw={"rmax": 0.5, "rstep": 0.1, "as_period": False}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12/2), layout="constrained")
plot_directional_spectrum(
    np.log10(F_ADCP), ax=ax1, levels=None, colorbar=False,
    axes_kw=axes_kw, vmin=vmin, vmax=vmax,
    wspd=U10[run_ind], wdir=winddir[run_ind],
    curspd=U_sfc_mag_m_s[run_ind],curdir=U_sfc_dir_deg[run_ind]
)


plot_directional_spectrum(
    np.log10(F_EPSS), ax=ax2, levels=None, colorbar=True,
    cbar_kw={"label": "$F(f,\\theta)\\;\\mathrm{[m^2 Hz^{-1} deg^{-1}]}$"},
    axes_kw=axes_kw, vmin=vmin, vmax=vmax,
    wspd=U10[run_ind], wdir=winddir[run_ind],
    curspd=U_sfc_mag_m_s[run_ind],curdir=U_sfc_dir_deg[run_ind]
)
_ = ax1.set(xlabel="", ylabel="", title="MEM, ADCP")
_ = ax2.set(xlabel="", ylabel="", title="MEM, E-PSS")


ax1.grid(False)
ax2.grid(False)
plt.savefig('../_figures/directional_spectra_polar.pdf',bbox_inches='tight')


# %%


D_ADCP.data[D_ADCP.data<1e-10] = np.nan
D_EPSS.data[D_EPSS.data<1e-10] = np.nan


winddir_plot = np.mod(winddir[run_ind]+180,360)
if winddir_plot > 180:
    winddir_plot = winddir_plot-360


Dlims = [0,0.012]


fig, (ax0, ax1, cax) = plt.subplots(ncols=3, figsize=(12, 5),gridspec_kw={"width_ratios":[1,1, 0.05]})


inds_keep = f_Hz_ADCP < f_cut_high
inds_keep_MEM = D_EPSS["frequency"].data < f_cut_high
inds_keep_Pyxis = Df_Pyxis["frequency"].data > f_cut_high*1.1


pc0 = ax0.pcolor(theta_deg_ADCP,f_Hz_ADCP[inds_keep],D_ADCP[inds_keep,:],vmin=Dlims[0],vmax=Dlims[1],cmap='viridis')
ax0.pcolor(Df_Pyxis["direction"],Df_Pyxis["frequency"].data[inds_keep_Pyxis],Df_Pyxis.data[inds_keep_Pyxis,:],vmin=Dlims[0],vmax=Dlims[1],cmap='viridis')
ax0.plot(winddir_plot*np.float64([1.0,1.0]),np.float64([1e-3,1e3]),color='red',label='wind direction')
ax0.set_yscale('log')
ax0.set_xticks(np.arange(-360,360,45))
ax0.set_xlim(-180,180)
ax0.set_ylim(1e-2,2e1)
ax0.set_xlabel(r'$\theta$ [$\circ$]')
ax0.set_ylabel('f [Hz]')
ax0.text(0.04,0.93,'(a)',color='white',fontsize=12,ha='center',va='center',transform=ax0.transAxes)
ax0.text(-37, 2.2e-2, 'MEM, ADCP',
         fontsize=12,
         color='black',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=1))


pc1 = ax1.pcolor(D_EPSS["direction"],D_EPSS["frequency"].data[inds_keep_MEM],D_EPSS.data[inds_keep_MEM,:]/2.0,vmin=Dlims[0],vmax=Dlims[1],cmap='viridis')
ax1.pcolor(Df_Pyxis["direction"],Df_Pyxis["frequency"].data[inds_keep_Pyxis],Df_Pyxis.data[inds_keep_Pyxis,:],vmin=Dlims[0],vmax=Dlims[1],cmap='viridis')
ax1.plot(winddir_plot*np.float64([1.0,1.0]),np.float64([1e-3,1e3]),color='red',label='wind direction')
ax1.set_yscale('log')
ax1.set_xticks(np.arange(-360,360,45))
ax1.set_xlim(-180,180)
ax1.set_ylim(1e-2,2e1)
ax1.set_xlabel(r'$\theta$ [$\circ$]')
ax1.set_yticklabels([])
ax1.text(0.04,0.93,'(b)',color='white',fontsize=12,ha='center',va='center',transform=ax1.transAxes)
ax1.text(-40, 2.2e-2, 'MEM, E-PSS',
         fontsize=12,
         color='black',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=1))


plt.subplots_adjust(right=1.05)


cbar = fig.colorbar(pc1, cax=cax)
cbar.set_label(r'$D(f,\theta)$')


plt.tight_layout()


plt.savefig('../_figures/directional_spectra_combined.pdf',bbox_inches='tight')


