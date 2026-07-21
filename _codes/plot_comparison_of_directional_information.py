# Plot comparison of directional wave information (MWD and directional spreading)
# between ADCP (MEM) and E-PSS (EWDM) estimates. Binned by U10.
# @author: nathanlaxague

import numpy as np
import xarray as xr

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

from eta_field_recon import lindisp_with_current
from subroutines.utils import (figure_style, compute_mean_wave_direction_and_spreading,
                               wind_speed_bins, binned_center_spread, write_tex_macros,
                               ewdm_low_cutoff, NUM_RUNS, WATER_DEPTH_M)
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")

panel_labels = ['(a)','(b)']

path = '../_data/'

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

Ff_EPSS = np.nan*np.ones((num_runs,num_f))

smoothnum = 5

theta_halfwidth = 120

f_cut_low = 0.08
f_cut_high = 0.3
f_cut_high_EPSS = 0.7          # E-PSS directional spreading trusted to higher f than MWD/Tm01

# trusted band for sigma_theta(f): faded below f_low (EWDM low-scale cutoff);
# ADCP solid only to f_ADCP_trust_high [Hz], E-PSS solid above
f_ADCP_trust_high = 0.3
_, f_low = ewdm_low_cutoff()
alpha_faded = 0.30             # opacity of each estimate beyond its trusted band

# f_m02 = sqrt(m2/m0) over [F_HP, F_LP] [Hz]; matches Tm02 figure band
F_HP, F_LP = 0.10, 0.7
def f_m02_of(freq, Sf):
    freq = np.asarray(freq, float)
    Sf = np.asarray(Sf, float)
    sh = (-1,) + (1,)*(Sf.ndim-1)
    band = ((freq >= F_HP) & (freq <= F_LP)).astype(float).reshape(sh)
    f2 = (freq**2).reshape(sh)
    return np.sqrt(np.nansum(band*f2*Sf, axis=0)/np.nansum(band*Sf, axis=0))
f_m02 = f_m02_of(f_Hz_omni, F_f_m2_Hz_omni)

SIGMA_HALFWIDTH = 90.0   # single-lobe sigma_theta window half-width [deg]; +-90 excludes the direct mirror lobe


def _smooth5(x, n=smoothnum):
    # centered n-point rolling median (NaN-aware, edge-shrinking) over the scale axis
    x = np.asarray(x, float)
    m = len(x)
    out = x.copy()
    for i in range(m):
        w = x[max(0, i-n//2):min(m, i+n//2+1)]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = np.median(w)
    return out


def lobe_spread(dirs_deg, dens, halfwidth=SIGMA_HALFWIDTH):
    # single-lobe RMS directional spread sigma_theta(scale); +-halfwidth deg isolates dominant lobe
    th = np.radians(np.asarray(dirs_deg))
    E = np.nan_to_num(dens)
    nsc = E.shape[0]
    sig = np.full(nsc, np.nan)
    h = np.radians(halfwidth)
    for j in range(nsc):
        D = E[j]
        s = D.sum()
        if s <= 0:
            continue
        a = (D*np.cos(th)).sum()
        b = (D*np.sin(th)).sum()
        d = np.angle(np.exp(1j*(th - np.arctan2(b, a))))
        Dk = D*(np.abs(d) <= h)
        sk = Dk.sum()
        if sk <= 0:
            continue
        thc = np.arctan2((Dk*np.sin(th)).sum(), (Dk*np.cos(th)).sum())
        d2 = np.angle(np.exp(1j*(th - thc)))
        sig[j] = np.degrees(np.sqrt((Dk*d2**2).sum()/sk))
    return sig


def lobe_sigma(F):
    # single-lobe sigma_theta(scale) from xarray (scale, direction) [deg direction coord]; smoothed
    return _smooth5(lobe_spread(F['direction'].data, np.nan_to_num(F.data)))


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
    Ff_ADCP = F_ADCP.integrate('direction')
    
    F_EPSS = ds_EPSS_spect['F_f_d'][:,:,run_ind]
    
    f_EPSS = F_EPSS['frequency'].data
    d_EPSS = F_EPSS['direction'].data
    
    Ff_EPSS[run_ind,:] = F_EPSS.integrate('direction')
    
    inds_exclude = (F_ADCP["frequency"].data > f_cut_high) | (F_ADCP["frequency"].data < f_cut_low)

    # spreading: ADCP high-cut only; MWD/Tm01 keep standard band [f_cut_low, f_cut_high]
    F_ADCP_spread = F_ADCP.copy(deep=True)
    F_ADCP_spread.data[F_ADCP_spread["frequency"].data > f_cut_high,:] = 0

    F_ADCP.data[inds_exclude,:] = 0

    inds_exclude = (F_EPSS["frequency"].data > f_cut_high) | (F_EPSS["frequency"].data < f_cut_low)
    F_EPSS.data[inds_exclude,:] = 0

    # MWD over ADCP-trusted band [f_cut_low, f_ADCP_trust_high]; direction deg CW-from-N
    F_EPSS_mwd = F_EPSS.copy(deep=True)
    F_EPSS_mwd.data[F_EPSS_mwd["frequency"].data > f_ADCP_trust_high,:] = 0
    mwd_EPSS, _ = compute_mean_wave_direction_and_spreading(F_EPSS_mwd,theta_halfwidth,smoothnum)
    MWD_EPSS[run_ind] = mwd_EPSS

    # sigma_theta(f) on extended band [f_cut_low, f_cut_high_EPSS]
    F_EPSS_spread = ds_EPSS_spect['F_f_d'][:,:,run_ind].copy(deep=True)
    inds_exclude_spread = (F_EPSS_spread["frequency"].data > f_cut_high_EPSS) | (F_EPSS_spread["frequency"].data < f_cut_low)
    F_EPSS_spread.data[inds_exclude_spread,:] = 0
    spread_EPSS = lobe_sigma(F_EPSS_spread)          # single-lobe (estimator-fair vs ADCP)
    SPREAD_EPSS[run_ind,:] = spread_EPSS
    
    total_energy = F_ADCP.integrate('frequency').integrate('direction')
    
    if total_energy > 0:
        F_ADCP_mwd = F_ADCP.copy(deep=True)
        F_ADCP_mwd.data[F_ADCP_mwd["frequency"].data > f_ADCP_trust_high,:] = 0
        mwd_ADCP, _ = compute_mean_wave_direction_and_spreading(F_ADCP_mwd,theta_halfwidth,smoothnum)
        spread_ADCP = lobe_sigma(F_ADCP_spread)      # single-lobe (estimator-fair vs E-PSS)

        Ff_ADCP = F_ADCP.integrate('direction').data
        f_m02_ADCP = f_m02_of(f_Hz_ADCP, Ff_ADCP)
        f_diff = np.abs(f_m02_ADCP-f_Hz_ADCP)
        ind = np.argmin(f_diff)
        
        ind_peak_ADCP[run_ind] = ind
        MWD_ADCP[run_ind] = mwd_ADCP
        SPREAD_ADCP[run_ind,:] = spread_ADCP
        
    else:
        ind_peak_ADCP[run_ind] = 0
        MWD_ADCP[run_ind] = np.nan
        SPREAD_ADCP[run_ind,:] = np.nan
        
    f_diff = np.abs(f_m02[run_ind]-f_Hz)
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

# Fold MWD_EPSS into [-90, 90] to resolve 180° ambiguity
inds_northerly = MWD_EPSS < -90
MWD_EPSS[inds_northerly] = MWD_EPSS[inds_northerly] + 180
inds_northerly = MWD_EPSS > 90
MWD_EPSS[inds_northerly] = MWD_EPSS[inds_northerly] - 180
MWD_diff = MWD_EPSS-MWD_ADCP_shifted

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

bin_medians, bin_mad, _, _ = binned_center_spread(U10_m_s, MWD_diff, U10_bin_edges)
bin_upper = bin_medians + bin_mad
bin_lower = bin_medians - bin_mad

fig = plt.figure(figsize=(fullwidth/2,fullwidth/2))
plt.fill_between(U10_bin_centers, bin_upper, bin_lower, color=color_list[2], alpha=0.25)
plt.plot(U10_bin_centers,bin_medians,'-',color=color_list[2],linewidth=2,label=r'$\theta_{E-PSS}-\theta_{ADCP}$')
plt.plot([0,16],[0,0],'--',color='gray')
plt.xlim(*U10_xlim)
plt.xticks(U10_xticks)
plt.xticks(np.arange(0,16,2))
plt.yticks(np.arange(-360,360,15))
plt.ylim(-45,45)
plt.xlabel(r'$U_{10}$ [m s$^{-1}$]')
plt.ylabel(r'$\Delta\theta_0$ [$\circ$]')

plt.savefig('../_figures/delta_theta_nought.pdf',bbox_inches='tight')


# %%

SPREAD_ADCP_peak = np.nan*np.ones(num_runs)
SPREAD_EPSS_peak = SPREAD_ADCP_peak.copy()

for run_num in np.arange(0,num_runs):
    SPREAD_ADCP_peak[run_num] = SPREAD_ADCP[run_num,ind_peak_ADCP[run_num]]
    SPREAD_EPSS_peak[run_num] = SPREAD_EPSS[run_num,ind_peak_EPSS[run_num]]

SPREAD_peak = np.nan*np.ones((num_runs,2))
SPREAD_peak[:,0] = SPREAD_ADCP_peak
SPREAD_peak[:,1] = SPREAD_EPSS_peak

# Bias, MAE, and RMSE for directional spreading at f_m02 (E-PSS minus ADCP)
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

labels = ['ADCP','E-PSS']

run_ind = 162
spread_ADCP = SPREAD_ADCP[run_ind,:]
spread_EPSS = SPREAD_EPSS[run_ind,:]

def plot_split(ax, x, y, lo, hi, label):
    # solid within [lo, hi], faded outside (hi=None = no upper bound)
    x = np.asarray(x)
    solid = np.ones(len(x), bool)
    if lo is not None: solid &= x >= lo
    if hi is not None: solid &= x <= hi
    line, = ax.plot(x, np.where(solid, y, np.nan), label=label, linewidth=2)
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
            ax.plot(x, np.where(fm, y, np.nan), color=line.get_color(), linewidth=2, alpha=alpha_faded)

fig,axs = plt.subplots(1,2,figsize=(fullwidth,fullwidth*0.4))
plot_split(axs[0], F_ADCP["frequency"], spread_ADCP, f_low, f_ADCP_trust_high, "ADCP")
plot_split(axs[0], F_EPSS["frequency"], spread_EPSS, f_low, None, "E-PSS")
axs[0].plot(f_Hz[ind_peak_EPSS[run_ind]]*np.float64([1.0,1.0]),[0,90])
axs[0].set_xscale('log')
axs[0].set_yticks(np.arange(0,360,15))
axs[0].set_ylim(0,90)
axs[0].set_xlim(1e-2,1e0)
axs[0].set_xlabel('f [Hz]')
axs[0].set_ylabel(r'$\sigma_{\theta}$ [$\circ$]')
axs[0].text(f_Hz[ind_peak_EPSS[run_ind]]*0.9,82.5,r'$f_{m02}$',color=color_list[2],ha='right')

axs[0].grid(which='major', linestyle='-', linewidth=0.75)
axs[0].grid(which='minor', linestyle=':', linewidth=0.75)

for n in np.arange(2):

    bin_medians, bin_mad, _, _ = binned_center_spread(U10_m_s, SPREAD_peak[:,n], U10_bin_edges)
    bin_upper = bin_medians + bin_mad
    bin_lower = bin_medians - bin_mad

    axs[1].fill_between(U10_bin_centers, bin_upper, bin_lower, color=color_list[n], alpha=0.25)
    axs[1].plot(U10_bin_centers,bin_medians,'-',linewidth=2,label=labels[n])


axs[1].set_yticks(np.arange(0,360,15))
axs[1].set_ylim(0,90)
axs[1].set_xlim(*U10_xlim)
axs[1].set_xticks(U10_xticks)
axs[1].set_xticks(np.arange(0,16,2))
axs[1].set_xlabel(r'$U_{10}$ [m s$^{-1}$]')
axs[1].set_ylabel(r'$\sigma_{\theta}$, evaluated at $f=f_{m02}$ [$\circ$]')
axs[1].legend()

for n in np.arange(2):
    
    axs[n].text(0.05,0.95,panel_labels[n],fontsize=fsize,ha='center',va='center',transform=axs[n].transAxes)
    
plt.savefig('../_figures/directional_spreading_comparison.pdf',bbox_inches='tight')





