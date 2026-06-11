# Plot slope distribution statistics (MSS and Gram-Charlier coefficients)
# binned by U10, compared against Cox & Munk [1954] and Breon & Henriot [2006].
# @author: nathanlaxague

import numpy as np
import xarray as xr

import netCDF4 as nc

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from subroutines.utils import *
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")

panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)']

path = '../_data/'

ds = nc.Dataset(path+'slope_statistics_dataset.nc')
slope_stats_output_names = ['c21','c03','c40','c04','c22','R_squared','RMSE']

ds_Elfouhaily = nc.Dataset(path+'Elfouhaily_et_al_1997_cumulative_mss.nc')

U10_m_s_all = ds['U10']

U10_m_s_vec = np.arange(0,15,0.1)

BH_slope_stats = xr.Dataset()
BH_slope_stats['U10'] = U10_m_s_vec
BH_slope_stats['c21'] = -9.1e-4*U10_m_s_vec**2
BH_slope_stats['c03'] = -0.45*(1+np.exp(7-U10_m_s_vec))**-1
BH_slope_stats['c40'] = np.zeros(len(U10_m_s_vec)) + 0.3
BH_slope_stats['c04'] = np.zeros(len(U10_m_s_vec)) + 0.4
BH_slope_stats['c22'] = np.zeros(len(U10_m_s_vec)) + 0.12

CM_slope_stats = xr.Dataset()
CM_slope_stats['U10'] = U10_m_s_vec
CM_slope_stats['c21'] = 0.01-8.6e-3*U10_m_s_vec
CM_slope_stats['c03'] = 0.04-0.033*U10_m_s_vec
CM_slope_stats['c40'] = np.zeros(len(U10_m_s_vec)) + 0.4
CM_slope_stats['c04'] = np.zeros(len(U10_m_s_vec)) + 0.23
CM_slope_stats['c22'] = np.zeros(len(U10_m_s_vec)) + 0.12

slope_stats_output_names_truncated = [r'$c_{21}$',r'$c_{03}$',r'$c_{40}$',r'$c_{04}$',r'$c_{22}$']
slope_stats_uncertainties = [1e-2,1e-2,0.05,0.1,0.03]

dolp_gain_choices = ['no gain','lab gain','empirical gain']

U10_bin_centers, U10_bin_edges, dU = wind_speed_bins()

def zero_centered_limit(env):
    # Symmetric y-limit about zero containing env (max |mean +/- 95% CI| over
    # all gains and bins), rounded up to the nearest major tick. Tick step is
    # drawn from the 1-2-5 family targeting ~6 ticks across the axis.
    step_target = env/3.0
    mag = 10.0**np.floor(np.log10(step_target))
    for m in (1, 2, 5, 10):
        step = m*mag
        if step >= step_target:
            break
    limit = np.ceil(env/step - 1e-9)*step
    return limit, step

#%% Upwind and crosswind mean square slope

particular_ind = 1; # corresponding to k_cut = 100 rad/m

mss_BH = np.nan*np.ones((len(U10_m_s_vec),2))
mss_BH[:,0] = 3e-3 + 1.85*1e-3*U10_m_s_vec
mss_BH[:,1] = 1e-3 + 3.16*1e-3*U10_m_s_vec

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(fullwidth, fullwidth/2))

titles = ['crosswind','upwind']
suffix_tags = ['c','u']

for i in np.arange(2):
    
    bin_upper = mss_BH[:,i] + 5e-4
    bin_lower = mss_BH[:,i] - 5e-4
    axs[1-i].fill_between(U10_m_s_vec, bin_upper, bin_lower, color='black', alpha=0.15)
    axs[1-i].plot(U10_m_s_vec,mss_BH[:,i],'-',linewidth=1,color='black',label='B & H [2006]',markersize=10)

    for j in np.arange(len(dolp_gain_choices)):
        
        inds = ds['DoLP gain'][:] == dolp_gain_choices[j]
        shortname = 'mss_' + suffix_tags[i]
        longname = shortname + '_long'
        all_values = np.ma.filled(ds[shortname][:], np.nan) + np.ma.filled(ds[longname][:], np.nan)
        values = all_values[inds]
        U10_m_s = np.ma.filled(U10_m_s_all[:], np.nan)[inds]

        # Drop NaN runs before binning so one NaN does not null an entire bin
        finite = np.isfinite(values) & np.isfinite(U10_m_s)
        values = values[finite]
        U10_m_s = U10_m_s[finite]

        bin_means, bin_edges, binnumber = stats.binned_statistic(U10_m_s,values, statistic='mean', bins=U10_bin_edges)
        bin_std, _, _ = stats.binned_statistic(U10_m_s,values, statistic='std', bins=U10_bin_edges)
        bin_counts, _, _ = stats.binned_statistic(U10_m_s, values, statistic='count', bins=U10_bin_edges)

        # 95% CI on the bin mean: standard error is std/sqrt(N), not std/N
        bin_95CI = 1.96*bin_std/np.sqrt(bin_counts)
        bin_upper = bin_means + bin_95CI
        bin_lower = bin_means - bin_95CI

        axs[1-i].fill_between(U10_bin_centers, bin_upper, bin_lower, color=color_list[j], alpha=0.25)
        axs[1-i].plot(U10_bin_centers,bin_means,'-',linewidth=2,label=dolp_gain_choices[j],markersize=10)
            
    axs[1-i].set_xlabel(r'$U_{10}$ [m s$^{-1}$]')
    axs[1-i].set_title(titles[i])
    axs[1-i].set_xlim(0,14)
    axs[1-i].set_ylim(0,0.045)
    

axs[0].plot(ds_Elfouhaily['U10'][:],ds_Elfouhaily['mss_u_block'][particular_ind],'--',color=[0,0,0.5],label=r'E97, $k<100$ rad m$^{-1}$')
axs[1].plot(ds_Elfouhaily['U10'][:],ds_Elfouhaily['mss_c_block'][particular_ind],'--',color=[0,0,0.5],label=r'E97, $k<100$ rad m$^{-1}$')
    
axs[0].set_ylabel('mss')
axs[1].legend(loc='upper left')

plt.tight_layout()

plt.savefig('../_figures/mss_upwind_crosswind.pdf',bbox_inches='tight')

#%% Gram-Charlier coefficients from least-squares fits to slope PDFs

# Grid placement (row, col) per coefficient and for the counts panel:
# c22 upper-left, counts upper-right; c40/c04 middle row; c21/c03 bottom row.
panel_positions = [(2,0),(2,1),(1,0),(1,1),(0,0)]  # c21,c03,c40,c04,c22
counts_position = (0,1)

fig, axs = plt.subplots(3, 2, sharex=True, figsize=(fullwidth, fullheight*0.9))

for i, varname in zip(np.arange(len(slope_stats_output_names_truncated)),slope_stats_output_names_truncated):

    all_values = ds[slope_stats_output_names[i]][:]

    env = 0.0  # max |mean +/- 95% CI| over all gains/bins, for the y-limit

    for j in np.arange(len(dolp_gain_choices)):

        inds = ds['DoLP gain'][:] == dolp_gain_choices[j]
        values = all_values[inds]
        U10_m_s = U10_m_s_all[inds]

        bin_means, bin_edges, binnumber = stats.binned_statistic(U10_m_s,values, statistic='mean', bins=U10_bin_edges)
        bin_std, _, _ = stats.binned_statistic(U10_m_s,values, statistic='std', bins=U10_bin_edges)
        bin_counts, _, _ = stats.binned_statistic(U10_m_s, values, statistic='count', bins=U10_bin_edges)

        # 95% CI on the bin mean: standard error is std/sqrt(N), not std/N
        bin_95CI = 1.96*bin_std/np.sqrt(bin_counts)
        bin_upper = bin_means + bin_95CI
        bin_lower = bin_means - bin_95CI

        env = np.nanmax([env, np.nanmax(np.abs(bin_upper)), np.nanmax(np.abs(bin_lower))])

        # Signal-to-noise of the empirical-gain bin means (flags unresolved coefficients)
        if dolp_gain_choices[j] == 'empirical gain':
            bin_SNR = np.abs(bin_means)/(bin_std/np.sqrt(bin_counts))

        row_index, col_index = panel_positions[i]

        axs[row_index,col_index].fill_between(U10_bin_centers, bin_upper, bin_lower, color=color_list[j], alpha=0.25)
        axs[row_index,col_index].plot(U10_bin_centers,bin_means,'-',linewidth=2,label=dolp_gain_choices[j],markersize=10)

    bin_upper = BH_slope_stats[slope_stats_output_names[i]] + slope_stats_uncertainties[i]
    bin_lower = BH_slope_stats[slope_stats_output_names[i]] - slope_stats_uncertainties[i]
    
    axs[row_index,col_index].fill_between(BH_slope_stats['U10'], bin_upper, bin_lower, color='black', alpha=0.15)
    axs[row_index,col_index].plot(BH_slope_stats['U10'],BH_slope_stats[slope_stats_output_names[i]],label='B & H [2006]',color='black',linewidth=2)
    axs[row_index,col_index].plot(CM_slope_stats['U10'],CM_slope_stats[slope_stats_output_names[i]],':',label='C & M [1954]',color='black',linewidth=2)
    axs[row_index,col_index].set_ylabel(varname)
    axs[row_index,col_index].set_xlim(0,14)
    ylimit, ytick = zero_centered_limit(env)
    axs[row_index,col_index].set_ylim(-ylimit, ylimit)
    axs[row_index,col_index].set_yticks(np.arange(-ylimit, ylimit + ytick/2, ytick))
    axs[row_index,col_index].set_xticks(np.arange(0,16,2))
    axs[row_index,col_index].text(0.05,0.95,panel_labels[2*row_index+col_index],fontsize=fsize,ha='center',va='center',zorder=7,transform=axs[row_index,col_index].transAxes)

    if row_index == 2:
        axs[row_index,col_index].set_xlabel(r'$U_{10}$ [m s$^{-1}$]')

    # Skewness coefficients (c21, c03) never reach 2-sigma significance in any
    # wind bin (empirical-gain SNR < 2 throughout); cover and mark as unresolved.
    if slope_stats_output_names[i] in ('c21','c03') and np.nanmax(bin_SNR[1::]) < 2.0:
        axs[row_index,col_index].axhline(0.0, color='0.5', linewidth=0.8, linestyle='--', zorder=0)
        axs[row_index,col_index].add_patch(plt.Rectangle((0,0),1,1,transform=axs[row_index,col_index].transAxes,
            facecolor='0.5', alpha=0.4, zorder=5))
        axs[row_index,col_index].text(0.5,0.05,'not resolved (SNR < 2)',fontsize=fsize,
            ha='center',va='bottom',style='italic',color='red',zorder=6,
            transform=axs[row_index,col_index].transAxes)

    if i == 4:
        axs[row_index,col_index].legend(loc='lower left')

row_index, col_index = counts_position

axs[row_index,col_index].bar(U10_bin_centers,bin_counts,color='black',label='counts',width=dU,alpha=0.5)

axs[row_index,col_index].set_ylabel('counts per bin')
axs[row_index,col_index].set_xlim(0,14)
axs[row_index,col_index].set_ylim(0,40)
axs[row_index,col_index].set_xticks(np.arange(0,16,2))
axs[row_index,col_index].text(0.05,0.95,panel_labels[2*row_index+col_index],fontsize=fsize,ha='center',va='center',transform=axs[row_index,col_index].transAxes)
    
plt.tight_layout()

plt.savefig('../_figures/slope_distribution_GC_coeffs.pdf',bbox_inches='tight')
