"""
Compare mean wave direction and directional spreading sigma_theta(f) across six
E-PSS directional estimators against the ADCP (IMLM) reference. Binned by U10.
@author: nathanlaxague
"""

import numpy as np
import xarray as xr

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

from eta_field_recon import lindisp_with_current
from subroutines.utils import (figure_style, compute_mean_wave_direction_and_spreading,
                               wind_speed_bins, binned_center_spread, write_tex_macros,
                               ewdm_low_cutoff, lobe_sigma, NUM_RUNS, WATER_DEPTH_M,
                               SPREAD_SMOOTHNUM, SPREAD_SMOOTHNUM_DIRECT)
project_colors,fullwidth,fullheight,fsize = figure_style()

# curve palette: black for the ADCP reference, then the project palette shared
# with plot_DoLP_AOI_inference, reordered onto the estimator roster below.
color_list = ['k',
              project_colors[2],        # EWDM Arrays, rust
              project_colors[4],        # EWDM Triplet, cerulean
              project_colors[0],        # IMLM, violet
              project_colors[3],        # EMEP, golden
              project_colors[1]]        # BDM, teal

import warnings

warnings.filterwarnings("ignore")

panel_labels = ['(a)','(b)']

# Estimator roster: F_f_d variable suffix -> legend label; '' is the base EWDM
# LSQ product. ADCP takes color 0 (black), estimators 1-5. Black is skipped in
# delta_theta_nought, where the ADCP is the reference rather than a curve.
# Order follows the panels of directional_spreading_panels, IMLM leading the
# traditional estimators as the one matching the ADCP's own inversion. Index 0
# must stay the base product: it is what the exported .tex macros summarize.
estimators = [('',              'EWDM Arrays'),
              ('_EWDM_triplet', 'EWDM Triplet'),
              ('_IMLM',         'IMLM'),
              ('_EMEP',         'EMEP'),
              ('_BDM',          'BDM')]
n_est = len(estimators)

path = '../_data/'
figpath = '../_figures/'

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
    
ds_EPSS_spect = xr.open_dataset(path+'ASIT2019_EPSS_directional_spectra.nc')

# convert direction axis to degrees for spreading estimator (F_f_d stays per-radian)
ds_EPSS_spect = ds_EPSS_spect.assign_coords(direction=np.degrees(ds_EPSS_spect['direction']))

ds_omnispect = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')
f_Hz_omni = ds_omnispect['frequency'][:].data
F_f_m2_Hz_omni = ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data

f_Hz_ADCP = ds_other['f_Hz_ADCP'][:]
theta_rad_ADCP = ds_other['theta_rad'][:]
Fftheta_m2_Hz_rad_ADCP = ds_other['F_f_theta_m2_Hz_rad_ADCP'][:]

U10_m_s = ds_other["U10_best"][:]
winddir_deg = ds_other["COARE_Wdir"][:]

theta_rad_ADCP = theta_rad_ADCP[np.arange(0,len(theta_rad_ADCP)-1)]

bigtheta = np.concatenate((theta_rad_ADCP-2*np.pi,theta_rad_ADCP,theta_rad_ADCP+2*np.pi),axis=0)*180/np.pi

inds_keep = (bigtheta >= -180) & (bigtheta <= 180)
theta_deg_ADCP = bigtheta[inds_keep]

f_Hz = ds_EPSS_spect['frequency'].data

num_runs = NUM_RUNS
num_f = len(f_Hz)

MWD_ADCP = np.nan*np.ones(num_runs)
MWD_EPSS = MWD_ADCP.copy()
Tm01_EPSS = MWD_ADCP.copy()
Tm01_ADCP = MWD_ADCP.copy()
ind_peak_ADCP = np.int16(np.ones(num_runs))
ind_peak_EPSS = ind_peak_ADCP.copy()
SPREAD_ADCP = np.nan*np.ones((num_runs,len(f_Hz_ADCP)))
SPREAD_EPSS = np.nan*np.ones((num_runs,num_f))

# per-estimator MWD and sigma_theta(f); index 0 (EWDM LSQ) feeds the macros
MWD_EST = np.nan*np.ones((n_est,num_runs))
SPREAD_EST = np.nan*np.ones((n_est,num_runs,num_f))

Ff_EPSS = np.nan*np.ones((num_runs,num_f))

smoothnum = 5

theta_halfwidth = 120

f_cut_low = 0.05
f_cut_high = 0.3
f_cut_high_EPSS = 0.7          # E-PSS directional spreading trusted to higher f than MWD/Tm01

# trusted band for sigma_theta(f): faded below f_low (EWDM low-scale cutoff);
# ADCP solid only to f_ADCP_trust_high [Hz], E-PSS solid above
f_ADCP_trust_high = 0.3
_, f_low = ewdm_low_cutoff()
alpha_faded = 0.30             # opacity of each estimate beyond its trusted band

# f_p over [F_HP, F_LP] [Hz] after Young (1995): int f E^q df / int E^q df. The
# E^q weighting concentrates the integral near the spectral maximum, so this is a
# peak frequency that does not depend on which single bin happens to be highest.
# Same band as the Tp figure.
F_HP, F_LP = 0.10, 0.7
PEAK_Q = 4
def f_p_of(freq, Sf, q=PEAK_Q):
    freq = np.asarray(freq, float)
    Sf = np.asarray(Sf, float)
    sh = (-1,) + (1,)*(Sf.ndim-1)
    band = ((freq >= F_HP) & (freq <= F_LP)).astype(float).reshape(sh)
    E = np.nan_to_num(band*Sf)**q
    num = np.nansum(freq.reshape(sh)*E, axis=0)
    den = np.nansum(E, axis=0)
    return np.where(den > 0, num/np.where(den > 0, den, 1.0), np.nan)
f_p = f_p_of(f_Hz_omni, F_f_m2_Hz_omni)


for run_ind in np.arange(0,num_runs):

    Fftheta_m2_Hz_rad_ADCP_particular = Fftheta_m2_Hz_rad_ADCP[:,:,run_ind].T
    
    Fftheta_m2_Hz_rad_ADCP_particular = Fftheta_m2_Hz_rad_ADCP_particular[:,np.arange(0,len(theta_deg_ADCP)-1)]
    
    bigFftheta = np.concatenate((Fftheta_m2_Hz_rad_ADCP_particular,Fftheta_m2_Hz_rad_ADCP_particular,Fftheta_m2_Hz_rad_ADCP_particular),axis=1)
        
    Fftheta_m2_Hz_rad_ADCP_particular = bigFftheta[:,inds_keep]
    
    # Build xarray dataset for ADCP directional spectrum
    dataset_ADCP = xr.Dataset(
        coords = {"frequency": f_Hz_ADCP, "direction": theta_deg_ADCP},
        data_vars = {
            "F_ADCP": (["frequency", "direction"], Fftheta_m2_Hz_rad_ADCP_particular*np.pi/180)
        }
    )
    
    F_ADCP = dataset_ADCP.F_ADCP
    # untruncated omnidirectional spectrum, kept for f_p alone: f_p is defined
    # over [F_HP, F_LP], which reaches past the f_cut_high applied to F_ADCP
    # below. Tm01 stays on the truncated band, matching the E-PSS side.
    Ff_ADCP_full = F_ADCP.integrate('direction').data

    F_EPSS = ds_EPSS_spect['F_f_d'][:,:,run_ind]
    
    f_EPSS = F_EPSS['frequency'].data
    d_EPSS = F_EPSS['direction'].data
    
    Ff_EPSS[run_ind,:] = F_EPSS.integrate('direction')
    
    inds_exclude = (F_ADCP["frequency"].data > f_cut_high) | (F_ADCP["frequency"].data < f_cut_low)

    # spreading: ADCP high-cut only; MWD/Tm01 keep standard band [f_cut_low, f_cut_high]
    F_ADCP_spread = F_ADCP.copy(deep=True)
    F_ADCP_spread.data[F_ADCP_spread["frequency"].data > f_cut_high,:] = 0

    F_ADCP.data[inds_exclude,:] = 0
    Ff_ADCP = F_ADCP.integrate('direction').data

    inds_exclude = (F_EPSS["frequency"].data > f_cut_high) | (F_EPSS["frequency"].data < f_cut_low)
    F_EPSS.data[inds_exclude,:] = 0

    # per estimator: MWD over [f_cut_low, f_ADCP_trust_high] (deg CW-from-N),
    # sigma_theta(f) over [f_cut_low, f_cut_high_EPSS]
    for n, (suffix, _lab) in enumerate(estimators):
        Fe = ds_EPSS_spect['F_f_d'+suffix][:,:,run_ind].copy(deep=True)
        fe = Fe["frequency"].data
        Fe.data[(fe > f_cut_high) | (fe < f_cut_low),:] = 0
        Fe_mwd = Fe.copy(deep=True)
        Fe_mwd.data[fe > f_ADCP_trust_high,:] = 0
        MWD_EST[n,run_ind], _ = compute_mean_wave_direction_and_spreading(
            Fe_mwd,theta_halfwidth,smoothnum)

        Fe_spread = ds_EPSS_spect['F_f_d'+suffix][:,:,run_ind].copy(deep=True)
        Fe_spread.data[(fe > f_cut_high_EPSS) | (fe < f_cut_low),:] = 0
        SPREAD_EST[n,run_ind,:] = lobe_sigma(Fe_spread, smooth=smoothnum)   # single-lobe (estimator-fair vs ADCP)

    MWD_EPSS[run_ind] = MWD_EST[0,run_ind]
    SPREAD_EPSS[run_ind,:] = SPREAD_EST[0,run_ind,:]
    
    total_energy = F_ADCP.integrate('frequency').integrate('direction')
    
    if total_energy > 0:
        F_ADCP_mwd = F_ADCP.copy(deep=True)
        F_ADCP_mwd.data[F_ADCP_mwd["frequency"].data > f_ADCP_trust_high,:] = 0
        mwd_ADCP, _ = compute_mean_wave_direction_and_spreading(F_ADCP_mwd,theta_halfwidth,smoothnum)
        spread_ADCP = lobe_sigma(F_ADCP_spread, smooth=smoothnum)      # single-lobe (estimator-fair vs E-PSS)

        f_p_ADCP = f_p_of(f_Hz_ADCP, Ff_ADCP_full)
        f_diff = np.abs(f_p_ADCP-f_Hz_ADCP)
        ind = np.argmin(f_diff)
        
        ind_peak_ADCP[run_ind] = ind
        MWD_ADCP[run_ind] = mwd_ADCP
        SPREAD_ADCP[run_ind,:] = spread_ADCP
        
    else:
        ind_peak_ADCP[run_ind] = 0
        MWD_ADCP[run_ind] = np.nan
        SPREAD_ADCP[run_ind,:] = np.nan
        
    f_diff = np.abs(f_p[run_ind]-f_Hz)
    f_diff[0] = 1e3
    ind = np.argmin(f_diff)
    ind_peak_EPSS[run_ind] = ind
    
    Ff_EPSS_val = F_EPSS.integrate('direction')
    Ff_EPSS_val = Ff_EPSS_val.data
    
    Tm01_ADCP[run_ind] = F_ADCP.integrate('direction').integrate('frequency')/np.trapezoid(f_Hz_ADCP*Ff_ADCP,x=f_Hz_ADCP)
    Tm01_EPSS[run_ind] = F_EPSS.integrate('direction').integrate('frequency')/np.trapezoid(F_EPSS['frequency'][:]*Ff_EPSS_val,x=F_EPSS['frequency'][:])

h_m_ADCP = 18.3
h_m_EPSS = WATER_DEPTH_M

omega_ADCP = 2*np.pi*Tm01_ADCP**-1
omega_EPSS = 2*np.pi*Tm01_EPSS**-1

C_m_s_disp_ADCP, _ = lindisp_with_current(omega_ADCP,h_m_ADCP,0)
C_m_s_disp_EPSS, _ = lindisp_with_current(omega_EPSS,h_m_EPSS,0)

# Refract ADCP MWD from 18.3 m to 15 m depth (coastline ~E-W)
MWD_ADCP_shifted = np.asin(C_m_s_disp_EPSS/C_m_s_disp_ADCP*np.sin(MWD_ADCP*np.pi/180))*180/np.pi

# Fold every estimator's MWD into [-90, 90] to resolve 180° ambiguity
MWD_EST = np.where(MWD_EST < -90, MWD_EST + 180, MWD_EST)
MWD_EST = np.where(MWD_EST >  90, MWD_EST - 180, MWD_EST)
MWD_EPSS = MWD_EST[0,:]
# estimator minus ADCP, one row per estimator
MWD_DIFF_EST = MWD_EST - MWD_ADCP_shifted[None,:]
MWD_diff = MWD_DIFF_EST[0,:]

# Bias, MAE, and RMSE for MWD (E-PSS minus ADCP)
MWD_bias = np.nanmean(MWD_diff)
MWD_mae = np.nanmean(np.abs(MWD_diff))
MWD_rmse = np.sqrt(np.nanmean(MWD_diff**2))

# %%

U10_bin_centers, U10_bin_edges, dU = wind_speed_bins()
# x-axis spans the wind-speed bins (XMAX = Umax + dU/2 = top bin edge)
U10_xlim = (U10_bin_centers[0] - dU/2, U10_bin_centers[-1] + dU/2)
# ticks at every bin edge, out to and including xmax (so the last tick is labeled)
U10_xticks = np.arange(U10_xlim[0], U10_xlim[1] + dU/2, dU)

fig = plt.figure(figsize=(fullwidth/2,fullwidth/2))
# one curve per estimator, each differenced against the ADCP; MAD bands are
# faint because five overlap
plotting_order = [1, 6, 5, 4, 3, 2]   # all above the grid zorder so curves sit over the grid
for n, (_suffix, lab) in enumerate(estimators):
    bin_medians, bin_mad, _, _ = binned_center_spread(U10_m_s, MWD_DIFF_EST[n,:], U10_bin_edges)
    plt.fill_between(U10_bin_centers, bin_medians+bin_mad, bin_medians-bin_mad,
                     color=color_list[n+1], alpha=0.12)
    plt.plot(U10_bin_centers,bin_medians,'-',color=color_list[n+1],linewidth=2,label=lab,zorder=plotting_order[n+1])
plt.plot([0,16],[0,0],'--',color='gray',zorder=plotting_order[0])
plt.gca().set_axisbelow(True)                 # draw the grid behind the curves
plt.xlim(*U10_xlim)
plt.xticks(U10_xticks)
plt.xticks(np.arange(0,16,2))
plt.yticks(np.arange(-360,360,15))
plt.ylim(-45,45)
plt.xlabel(r'$U_{10}$ [m s$^{-1}$]')
plt.ylabel(r'$\Delta\theta_0$ [$\circ$]')
plt.legend(ncol=2,fontsize=fsize-2,loc='upper right')

plt.savefig(figpath + 'delta_theta_nought.pdf',bbox_inches='tight')


# %%

SPREAD_ADCP_peak = np.nan*np.ones(num_runs)
SPREAD_EST_peak = np.nan*np.ones((n_est,num_runs))

for run_num in np.arange(0,num_runs):
    SPREAD_ADCP_peak[run_num] = SPREAD_ADCP[run_num,ind_peak_ADCP[run_num]]
    SPREAD_EST_peak[:,run_num] = SPREAD_EST[:,run_num,ind_peak_EPSS[run_num]]

SPREAD_EPSS_peak = SPREAD_EST_peak[0,:]
# column 0 ADCP, columns 1..n_est estimators; matches palette order
SPREAD_peak = np.column_stack([SPREAD_ADCP_peak, SPREAD_EST_peak.T])

# Bias, MAE, and RMSE for directional spreading at f_p (E-PSS minus ADCP)
SPREAD_diff = SPREAD_EPSS_peak - SPREAD_ADCP_peak
SPREAD_bias = np.nanmean(SPREAD_diff)
SPREAD_mae = np.nanmean(np.abs(SPREAD_diff))
SPREAD_rmse = np.sqrt(np.nanmean(SPREAD_diff**2))

# Export the directional comparison metrics as LaTeX macros for paper.tex
write_tex_macros('directional_values.tex', {
    'MwdBias':    f'{MWD_bias:.2f}',
    'MwdMAE':     f'{MWD_mae:.2f}',
    'MwdRMSE':    f'{MWD_rmse:.2f}',
    'SpreadBias': f'{SPREAD_bias:.2f}',
    'SpreadMAE':  f'{SPREAD_mae:.2f}',
    'SpreadRMSE': f'{SPREAD_rmse:.2f}',
}, source='plot_comparison_of_directional_information.py')

labels = ['ADCP'] + [lab for _suffix, lab in estimators]

run_ind = 162
spread_ADCP = SPREAD_ADCP[run_ind,:]

# sigma_theta(f) from the direct 3-D-FFT short-wave spectrum, this case only.
# sigma_theta depends on D alone, so the direct SLOPE spectrum needs no k^-2
# conversion. Its frequency grid is ~4x denser than the estimators', so the
# rolling median widens by the same ratio to span the same fractional bandwidth.
ds_direct = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
S_f_theta_direct = np.ma.filled(ds_direct['S_f_theta'][run_ind],np.nan).T   # (f, theta)
f_Hz_direct = np.asarray(ds_direct['f_Hz'][:])
theta_deg_direct = np.degrees(np.asarray(ds_direct['theta_rad'][:]))
ds_direct.close()

smoothnum_direct = smoothnum*SPREAD_SMOOTHNUM_DIRECT//SPREAD_SMOOTHNUM
spread_direct = lobe_sigma(
    xr.DataArray(S_f_theta_direct,dims=('frequency','direction'),
                 coords={'frequency': f_Hz_direct,'direction': theta_deg_direct}),
    smooth=smoothnum_direct)
f_direct_trust_low = 0.7        # the splice cut: direct is the reference above it

def plot_split(ax, x, y, lo, hi, label, color=None, linestyle='-',linewidth=2):
    # solid within [lo, hi], faded outside (hi=None = no upper bound)
    x = np.asarray(x)
    solid = np.ones(len(x), bool)
    if lo is not None: solid &= x >= lo
    if hi is not None: solid &= x <= hi
    line, = ax.plot(x, np.where(solid, y, np.nan), label=label, linewidth=linewidth,
                    color=color, linestyle=linestyle)
    if solid.any():
        i0, i1 = np.where(solid)[0][[0, -1]]
        faded = []
        if i0 > 0:
            m = np.zeros(len(x), bool)
            m[:i0+1] = True
            faded.append(m)
        if i1 < len(x)-1:
            m = np.zeros(len(x), bool)
            m[i1:] = True
            faded.append(m)
        for fm in faded:
            ax.plot(x, np.where(fm, y, np.nan), color=line.get_color(), linewidth=linewidth, alpha=alpha_faded, linestyle=linestyle)

fig,axs = plt.subplots(1,2,figsize=(fullwidth,fullwidth*0.4))
axs[0].text(2.8e-2,47,'ADCP',color='black',
            ha='center',va='bottom',fontsize=fsize-2)
plot_split(axs[0], F_ADCP["frequency"], spread_ADCP, f_low, f_ADCP_trust_high,
           "ADCP", color_list[0], linestyle='--', linewidth=2.5)
for n, (_suffix, lab) in enumerate(estimators):
    plot_split(axs[0], f_Hz, SPREAD_EST[n,run_ind,:], f_low, None, lab,
               color_list[n+1])
# the direct short-wave estimate, solid only above the splice cut, where it is
# the reference the estimators are read against rather than an extrapolation.
# Gray and dash-dot to read as a reference alongside the dashed black ADCP,
# without taking a slot in the estimator palette.
plot_split(axs[0], f_Hz_direct, spread_direct, f_direct_trust_low, None,
           'direct FFT', 'black', linestyle=':', linewidth=2.5)
# panel (a) carries no legend (panel (b) holds it), so this one is named inline
_i_lab = np.argmin(np.abs(f_Hz_direct-2.0))
axs[0].text(2.0,spread_direct[_i_lab]+2.5,'direct FFT',color='black',
            ha='center',va='bottom',fontsize=fsize-2)
# f_p marker black dotted, matching the ADCP curve
axs[0].plot(f_Hz[ind_peak_EPSS[run_ind]]*np.float64([1.0,1.0]),[0,90],
            '-',color='k',linewidth=1.5)
axs[0].set_xscale('log')
axs[0].set_yticks(np.arange(0,360,15))
axs[0].set_ylim(0,90)
axs[0].set_xlim(2e-2,5e0)
axs[0].set_xlabel('f [Hz]')
axs[0].set_ylabel(r'$\sigma_{\theta}$ [$\circ$]')
axs[0].text(f_Hz[ind_peak_EPSS[run_ind]]*0.9,82.5,r'$f_p$',color='k',ha='right')

axs[0].grid(which='major', linestyle='-', linewidth=0.75)
axs[0].grid(which='minor', linestyle=':', linewidth=0.75)
axs[0].set_axisbelow(True)                     # draw the grid behind the curves

for n in np.arange(SPREAD_peak.shape[1]):

    bin_medians, bin_mad, _, _ = binned_center_spread(U10_m_s, SPREAD_peak[:,n], U10_bin_edges)
    bin_upper = bin_medians + bin_mad
    bin_lower = bin_medians - bin_mad

    axs[1].fill_between(U10_bin_centers, bin_upper, bin_lower, color=color_list[n], alpha=0.12)
    axs[1].plot(U10_bin_centers,bin_medians,'--' if n==0 else '-',color=color_list[n],linewidth=2.5 if n==0 else 2,label=labels[n],zorder=plotting_order[n])


axs[1].set_axisbelow(True)                      # draw the grid behind the curves
axs[1].set_yticks(np.arange(0,360,15))
axs[1].set_ylim(0,90)
axs[1].set_xlim(*U10_xlim)
axs[1].set_xticks(U10_xticks)
axs[1].set_xticks(np.arange(0,16,2))
axs[1].set_xlabel(r'$U_{10}$ [m s$^{-1}$]')
axs[1].set_ylabel(r'$\sigma_{\theta}$, evaluated at $f=f_p$ [$\circ$]')
axs[1].legend(ncol=2,fontsize=fsize-2)

for n in np.arange(2):
    
    axs[n].text(0.05,0.95,panel_labels[n],fontsize=fsize,ha='center',va='center',transform=axs[n].transAxes)
    
plt.savefig(figpath + 'directional_spreading_comparison.pdf',bbox_inches='tight')





