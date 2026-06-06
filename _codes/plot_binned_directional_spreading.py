# Plot E-PSS directional spreading sigma_theta vs wave scale (f, k), binned by
# U10. EWDM (trusted below a per-axis boundary) and direct full-frame slope spectra
# (trusted above) drawn solid on their trusted side and faded beyond, as in
# plot_binned_omnispect. ADCP underlaid dashed (solid within [f_low, 0.25 Hz], faded
# beyond), mapped to k by deep-water dispersion (k = (2 pi f)^2 / g).
# @author: nathanlaxague

import numpy as np
import xarray as xr

import netCDF4 as nc

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from subroutines.utils import figure_style, compute_mean_wave_direction_and_spreading, wind_speed_bins
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")

g = 9.81

panel_labels = ['(a)','(b)']

path = '../_data/'


def band_average(A, n=5):
    # n-point running mean over the spectral (last) axis, NaN-aware
    out = np.full(A.shape, np.nan)
    half = n//2
    for j in range(A.shape[1]):
        lo = max(0, j-half); hi = min(A.shape[1], j+half+1)
        out[:,j] = np.nanmean(A[:,lo:hi], axis=1)
    return out


# spreading-estimator parameters (match plot_comparison_of_directional_information)
theta_halfwidth = 120
smoothnum = 5


def lobe_spread(directions_deg, density, halfwidth=90.0):
    # Per-scale downwave-lobe direction and spread for 180-deg-symmetric direct-slope
    # spectra: center on the resultant direction, keep energy within +-halfwidth,
    # take that lobe's centroid and rms angular width (excludes the upwind lobe; the
    # +-90 window bounds sigma_theta <= 90). Returns direction (deg CW from N),
    # spread (deg), resultant length.
    th = np.radians(np.asarray(directions_deg))
    cos_t, sin_t = np.cos(th), np.sin(th)
    hw = np.radians(halfwidth)
    E = np.nan_to_num(density)
    nsc = E.shape[0]
    th0 = np.full(nsc, np.nan); sig = np.full(nsc, np.nan); m1 = np.full(nsc, np.nan)
    for j in range(nsc):
        D = E[j]; s = D.sum()
        if s <= 0:
            continue
        a = (D*cos_t).sum(); b = (D*sin_t).sum()
        m1[j] = np.hypot(a, b)/s
        d = np.angle(np.exp(1j*(th - np.arctan2(b, a))))     # angle from dominant lobe
        Dk = D*(np.abs(d) <= hw); sk = Dk.sum()
        if sk <= 0:
            continue
        thc = np.arctan2((Dk*sin_t).sum(), (Dk*cos_t).sum())  # kept-lobe centroid
        d2 = np.angle(np.exp(1j*(th - thc)))
        th0[j] = np.degrees(thc)
        sig[j] = np.degrees(np.sqrt((Dk*d2**2).sum()/sk))
    return th0, sig, m1


def spreading(scale_values, directions, density):
    # sigma_theta vs scale from a directional spectrum density(scale, direction).
    # Scale axis labeled 'frequency' so the shared estimator applies to f and k;
    # per-scale spread is independent of that label.
    if not np.isfinite(density).any() or np.nansum(np.abs(density)) <= 0:
        return np.full(len(scale_values), np.nan)
    F = xr.Dataset(
        coords = {'frequency': scale_values, 'direction': directions},
        data_vars = {'F': (['frequency','direction'], np.nan_to_num(density))},
    ).F
    try:
        _, spread = compute_mean_wave_direction_and_spreading(F, theta_halfwidth, smoothnum)
        return spread.data
    except Exception:
        return np.full(len(scale_values), np.nan)


# E-PSS directional spectra (elevation) in frequency and wavenumber
ds_EPSS = xr.open_dataset(path+'ASIT2019_EPSS_directional_spectra.nc')
# Dataset direction is radians; the spreading estimator wraps in degrees (shape only,
# unaffected by the per-radian density level), so view the axis in degrees.
ds_EPSS = ds_EPSS.assign_coords(direction=np.degrees(ds_EPSS['direction']))
f_EPSS = ds_EPSS['frequency'].data
k_EPSS = ds_EPSS['wavenumber'].data
dir_EPSS = ds_EPSS['direction'].data

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
U10_m_s = ds_other['COARE_U10'][:]

# Direct full-frame slope directional spectra (slope^2 density; runs, dir, scale)
ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
f_direct = ds_emp['f_Hz'][:]
k_direct = ds_emp['k_rad_m'][:]
dir_direct = np.degrees(ds_emp['theta_rad'][:])      # deg CW from N (same sense as EWDM)

# ADCP directional spectrum; wrap directions onto [-180, 180] (per plot_comparison)
f_Hz_ADCP = ds_other['f_Hz_ADCP'][:]
theta_rad_ADCP = ds_other['theta_rad'][:]
Fftheta_ADCP = ds_other['F_f_theta_m2_Hz_rad_ADCP'][:]   # (theta, f, run)

theta_rad_ADCP = theta_rad_ADCP[np.arange(0,len(theta_rad_ADCP)-1)]
bigtheta = np.concatenate((theta_rad_ADCP-2*np.pi,theta_rad_ADCP,theta_rad_ADCP+2*np.pi),axis=0)*180/np.pi
inds_keep = (bigtheta >= -180) & (bigtheta <= 180)
theta_deg_ADCP = bigtheta[inds_keep]

num_runs = 190

SPREAD_EPSS_f = np.nan*np.ones((num_runs,len(f_EPSS)))
SPREAD_EPSS_k = np.nan*np.ones((num_runs,len(k_EPSS)))
SPREAD_ADCP_f = np.nan*np.ones((num_runs,len(f_Hz_ADCP)))

SPREAD_DIRECT_f = np.nan*np.ones((num_runs,len(f_direct)))
SPREAD_DIRECT_k = np.nan*np.ones((num_runs,len(k_direct)))


def direct_dir(dirs, dens):
    # direct-spectrum downwave-lobe directional spread (sigma_theta) per scale
    _, sp, _ = lobe_spread(dirs, dens, halfwidth=90.0)
    return sp


for run_ind in np.arange(0,num_runs):

    SPREAD_EPSS_f[run_ind,:] = spreading(f_EPSS, dir_EPSS, ds_EPSS['F_f_d'][:,:,run_ind].data)
    SPREAD_EPSS_k[run_ind,:] = spreading(k_EPSS, dir_EPSS, ds_EPSS['F_k_d'][:,:,run_ind].data)

    # direct full-frame slope spectra (transpose to scale-by-direction)
    SPREAD_DIRECT_f[run_ind,:] = direct_dir(dir_direct, np.nan_to_num(ds_emp['S_f_theta'][run_ind].T))
    SPREAD_DIRECT_k[run_ind,:] = direct_dir(dir_direct, np.nan_to_num(ds_emp['S_k_theta'][run_ind].T))

    # ADCP: triple-wrap the directions, slice to [-180, 180], spread vs f
    Ff = Fftheta_ADCP[:,:,run_ind].T                          # (f, theta)
    Ff = Ff[:,np.arange(0,len(theta_deg_ADCP)-1)]
    bigFf = np.concatenate((Ff,Ff,Ff),axis=1)
    Ff = bigFf[:,inds_keep]
    SPREAD_ADCP_f[run_ind,:] = spreading(f_Hz_ADCP, theta_deg_ADCP, Ff*np.pi/180)

# Bin by U10 (nanmean so a single NaN run does not drop a wind bin)
U_centers, U_boundaries, dU = wind_speed_bins()

def wind_bin(A):
    # mean over runs in each U10 bin (nanmean so one NaN run does not drop a bin)
    out = np.nan*np.ones((len(U_centers), A.shape[1]))
    for i in np.arange(len(U_centers)):
        inds = (U10_m_s > U_centers[i] - dU/2) & (U10_m_s <= U_centers[i] + dU/2)
        out[i,:] = np.nanmean(A[inds,:], axis=0)
    return out


def cap90(A):
    # drop unphysical spreads (> 90 deg) before binning
    return np.where(A > 90.0, np.nan, A)


SPREAD_EPSS_f_binned = band_average(wind_bin(cap90(SPREAD_EPSS_f)))
SPREAD_EPSS_k_binned = band_average(wind_bin(cap90(SPREAD_EPSS_k)))
SPREAD_ADCP_f_binned = band_average(wind_bin(cap90(SPREAD_ADCP_f)))
SPREAD_DIRECT_f_binned = band_average(wind_bin(cap90(SPREAD_DIRECT_f)))
SPREAD_DIRECT_k_binned = band_average(wind_bin(cap90(SPREAD_DIRECT_k)))

# Direct directional spectra above 7.5 Hz are untrustworthy; drop them.
f_cut = 7.5
SPREAD_DIRECT_f_binned[:, f_direct > f_cut] = np.nan

# Deep-water linear dispersion: map the ADCP frequency axis to k
k_ADCP = (2*np.pi*f_Hz_ADCP)**2/g

# ADCP directional estimate trusted within [f_low, f_ADCP_trust_high] (swell/wave
# band); faded beyond, as in plot_comparison_of_directional_information.
f_ADCP_trust_high = 0.25

cmap = plt.get_cmap('cividis')
colors = [cmap(j) for j in np.linspace(0,1,len(U_centers))]

# Horizontal axis limits (full range from plot_binned_omnispect)
f_lims = [1e-2,2e1]
k_lims = [1e-2,1e3]

lw_thick = 2.5
lw_thin = 1.5

# E-PSS reliability boundaries (per axis; from plot_binned_omnispect): EWDM trusted
# at/below the boundary, direct slope spectra at/above; each faded beyond.
f_bound = 0.7
k_bound = 2.0
alpha_faded = 0.30

# Low-scale EWDM cutoff: below this the EWDM elevation spectrum overshoots the Riegl
# lidar (1/k^2 slope-noise gain) to energy SNR < 0.5; fade the EWDM below it. From
# lambda = 73*L_FOV via finite-depth dispersion (h = 15 m).
h_m = 15.0
L_FOV_m = 2.915
n_frame_low = 73            # lambda/L_FOV at energy SNR = 0.5 (EWDM vs Riegl lidar)
k_low = 2*np.pi/(n_frame_low*L_FOV_m)
omega_low = np.sqrt(g*k_low*np.tanh(k_low*h_m))
f_low = omega_low/(2*np.pi)


def trusted_segments(x, bound, is_ewdm, low_bound=None):
    # solid (trusted) mask and faded (beyond) masks, each overlapping the solid
    # region by one point. EWDM trusted within [low_bound, bound]; direct slope
    # at/above bound.
    x = np.asarray(x)
    if is_ewdm:
        solid = (x >= (low_bound if low_bound is not None else -np.inf)) & (x <= bound)
    else:
        solid = x >= bound
    faded = []
    if solid.any():
        i0, i1 = np.where(solid)[0][[0, -1]]
        if i0 > 0:
            m = np.zeros(len(x), bool); m[:i0+1] = True; faded.append(m)
        if i1 < len(x)-1:
            m = np.zeros(len(x), bool); m[i1:] = True; faded.append(m)
    return solid, faded


# ADCP dashed estimate: solid within [f_low, f_ADCP_trust_high], faded beyond. The
# mask indexes f_Hz_ADCP; it applies to the k-mapped axis by position.
adcp_solid, adcp_faded = trusted_segments(f_Hz_ADCP, f_ADCP_trust_high, True, f_low)


# Per-column axis spec: (EWDM x, direct x, ADCP x, boundary, EWDM low bound, x-limits, x-label)
columns = [
    (f_EPSS,  f_direct,  f_Hz_ADCP, f_bound,  f_low,  f_lims,  'f [Hz]'),
    (k_EPSS,  k_direct,  k_ADCP,    k_bound,  k_low,  k_lims,  r'k [rad m$^{-1}$]'),
]

spread_EWDM = [SPREAD_EPSS_f_binned, SPREAD_EPSS_k_binned]
spread_DIRECT = [SPREAD_DIRECT_f_binned, SPREAD_DIRECT_k_binned]


def draw_technique(ax, x, Y, bound, is_ewdm, low_bound=None):
    # black-outlined solid on the trusted side of the boundary, faded beyond
    x = np.asarray(x)
    solid, faded = trusted_segments(x, bound, is_ewdm, low_bound)
    for i in np.arange(len(U_centers)):
        ax.semilogx(x[solid], Y[i,solid], color='black', linewidth=lw_thick)
    for i in np.arange(len(U_centers)):
        ax.semilogx(x[solid], Y[i,solid], color=colors[i], linewidth=lw_thin)
        for fm in faded:
            ax.semilogx(x[fm], Y[i,fm], color=colors[i], linewidth=lw_thin, alpha=alpha_faded)


def draw_panel(ax, x_EWDM, Y_EWDM, x_DIRECT, Y_DIRECT, x_ADCP, Y_ADCP, bound, low_bound):
    x_ADCP = np.asarray(x_ADCP)                               # ADCP dashed: solid in band, faded beyond
    for i in np.arange(len(U_centers)):
        ax.semilogx(x_ADCP[adcp_solid], Y_ADCP[i,adcp_solid], '--', color=colors[i], linewidth=lw_thin)
        for fm in adcp_faded:
            ax.semilogx(x_ADCP[fm], Y_ADCP[i,fm], '--', color=colors[i], linewidth=lw_thin, alpha=alpha_faded)
    draw_technique(ax, x_EWDM, Y_EWDM, bound, True, low_bound)  # EWDM, trusted in [low, bound]
    draw_technique(ax, x_DIRECT, Y_DIRECT, bound, False)        # direct slope, trusted above
    ax.axvline(bound, color='dimgray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(low_bound, color='dimgray', linestyle='--', linewidth=1, alpha=0.7)
    ax.grid(which='major', linestyle='-', linewidth=0.75)
    ax.grid(which='minor', linestyle=':', linewidth=0.75)


fig, axs = plt.subplots(1, 2, figsize=(fullwidth, fullwidth*0.5))

for n, (x_EWDM, x_DIRECT, x_ADCP, bound, low_bound, xlims, xlabel) in enumerate(columns):

    # directional spreading sigma_theta
    draw_panel(axs[n], x_EWDM, spread_EWDM[n], x_DIRECT, spread_DIRECT[n], x_ADCP, SPREAD_ADCP_f_binned, bound, low_bound)
    axs[n].set_xlim(xlims); axs[n].set_ylim(0,90); axs[n].set_yticks(np.arange(0,91,15))
    axs[n].set_xlabel(xlabel)
    axs[n].text(0.05,0.92,panel_labels[n],fontsize=fsize,ha='center',va='center',transform=axs[n].transAxes)

axs[0].set_ylabel(r'$\sigma_{\theta}$ [$^\circ$]')
axs[1].set_yticklabels([])

# E-PSS (solid: EWDM below boundary, direct slope above) vs ADCP (dashed) legend
axs[0].plot([],[],'-',color='black',linewidth=lw_thin,label='E-PSS (EWDM / direct)')
axs[0].plot([],[],'--',color='black',linewidth=lw_thin,label='ADCP')
axs[0].legend(loc='lower left',fontsize=fsize)

plt.tight_layout()

norm = BoundaryNorm(U_boundaries, cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axs, location='right', orientation='vertical', fraction=0.02, pad=0.02, aspect=40)
cbar.set_label(r'U$_{10}$ [m s$^{-1}$]')

plt.savefig('../_figures/directional_spreading_binned_by_wind.pdf',bbox_inches='tight')
