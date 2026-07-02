# Plot omnidirectional elevation and saturation spectra in frequency and
# wavenumber, binned by U10. Two techniques (integrated EWDM directional spectra
# and full-frame slope spectra, slope-to-elevation per axis) are drawn solid on
# their trusted side of a per-axis boundary and faded beyond. Saturation panels
# use the Bjorkqvist et al. (2019, JGR) dimensionless forms ((2pi f)^5 g^-2 F(f),
# k^3 F(k)) and universal constants (2.5e-2, 5.8e-3).
# @author: nathanlaxague

import os
import numpy as np
import xarray as xr

import netCDF4 as nc

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from eta_field_recon import lindisp_with_current
from subroutines.utils import figure_style, wind_speed_bins, ewdm_low_cutoff, WATER_DEPTH_M
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")


def band_average(A, n=5):
    # n-point running mean over the spectral (last) axis, NaN-aware
    out = np.full(A.shape, np.nan)
    half = n//2
    for j in range(A.shape[1]):
        lo = max(0, j-half)
        hi = min(A.shape[1], j+half+1)
        out[:,j] = np.nanmean(A[:,lo:hi], axis=1)
    return out


panel_labels = ['(a)','(b)','(c)','(d)']

path = '../_data/'

ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')

ds_EWDM = xr.open_dataset(path+os.environ.get('EPSS_EWDM_FILE', 'ASIT2019_EPSS_directional_spectra.nc'))

ds_other = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')

U10_m_s = ds_other["U10_best"]

h_m = WATER_DEPTH_M
g = 9.81

# Full-frame slope (direct) spectra; S_f and S_k are SLOPE spectra
f_Hz_slope = ds_emp['f_Hz'][:].data
k_rad_m_slope = ds_emp['k_rad_m'][:]
theta_rad = ds_emp['theta_rad'][:]
S_f_theta = ds_emp['S_f_theta'][:]
S_k_theta = ds_emp['S_k_theta'][:]

S_k_theta[:,:,len(k_rad_m_slope)-1] = np.nan

dtheta = np.median(np.diff(theta_rad))

S_f_slope = np.sum(S_f_theta,axis=1)*dtheta
S_k_slope = np.sum(np.reshape(k_rad_m_slope,(1,1,len(k_rad_m_slope)))*S_k_theta,axis=1)*dtheta

# Integrated EWDM elevation spectra
f_ewdm = ds_EWDM['frequency'].values
k_ewdm = ds_EWDM['wavenumber'].values
S_f_ewdm = ds_EWDM['S_f'].values       # (frequency, run)
F_k_ewdm = ds_EWDM['F_k'].values       # (wavenumber, run)

# EWDM/direct boundary: f_bound set below by level match in [0.30, 0.50] Hz (<=0.6 Hz fallback).
# f_high: gravity-band ceiling; EWDM tail faded only through f_high in the frequency column.
k_bound = k_rad_m_slope[0]
f_high = 1.0

# Low-scale EWDM trust cutoff; fade the EWDM below it
k_low, f_low = ewdm_low_cutoff()

U_centers, U_boundaries, dU = wind_speed_bins()

S_f_ewdm_binned = np.nan*np.ones((len(U_centers),len(f_ewdm)))
F_k_ewdm_binned = np.nan*np.ones((len(U_centers),len(k_ewdm)))
S_f_slope_binned = np.nan*np.ones((len(U_centers),len(f_Hz_slope)))
S_k_slope_binned = np.nan*np.ones((len(U_centers),len(k_rad_m_slope)))

dolp_gain_choices = ['no gain','lab gain','empirical gain']

for i in np.arange(len(U_centers)):

    inds = (U10_m_s > U_centers[i] - dU/2) & (U10_m_s <= U_centers[i] + dU/2)
    S_f_ewdm_binned[i,:] = np.nanmean(S_f_ewdm[:,inds],axis=1)
    F_k_ewdm_binned[i,:] = np.nanmean(F_k_ewdm[:,inds],axis=1)
    S_f_slope_binned[i,:] = np.nanmean(S_f_slope[inds,:],axis=0)
    S_k_slope_binned[i,:] = np.nanmean(S_k_slope[inds,:],axis=0)

# Light band-average of the (choppy at high k) EWDM spectra over the spectral axis
n_smooth = 5
S_f_ewdm_binned = band_average(S_f_ewdm_binned, n_smooth)
F_k_ewdm_binned = band_average(F_k_ewdm_binned, n_smooth)

# Linear dispersion for the frequency saturation/elevation weighting (h=15 m)
C_m_s_disp, k_ewdm_disp = lindisp_with_current(2*np.pi*f_ewdm,h_m,0)
# Gravity-only k(f) for direct-slope -> elevation; omits surface tension to avoid
# capillary k(f) shrinkage inflating F(f) at high f (omega^2 = g k tanh(k h)).
_w_slope = 2*np.pi*f_Hz_slope
k_rad_m_slope_disp = np.where(_w_slope > 0, _w_slope**2/g, 0.0)
for _ in range(80):
    k_rad_m_slope_disp = np.where(_w_slope > 0,
        _w_slope**2/(g*np.tanh(k_rad_m_slope_disp*h_m)), 0.0)

# Frequency: EWDM is elevation; direct slope -> elevation via /k(f)^2
F_f_ewdm_binned = S_f_ewdm_binned
F_f_slope_binned = np.reshape(k_rad_m_slope_disp,(1,len(k_rad_m_slope_disp)))**-2*S_f_slope_binned

# f_bound: frequency where EWDM and direct elevation spectra agree best in level
# (geometric-median ratio over wind bins), searched in [fmin, fmax].
def _frequency_handoff(fmin=0.30, fmax=0.50, fallback=0.60):
    cand = f_ewdm[(f_ewdm >= fmin) & (f_ewdm <= fmax)]
    best, best_score = fallback, np.inf
    for ff in cand:
        ew = F_f_ewdm_binned[:, int(np.argmin(np.abs(f_ewdm - ff)))]
        sl = np.array([np.interp(ff, f_Hz_slope, F_f_slope_binned[i]) for i in range(len(U_centers))])
        ok = np.isfinite(ew) & np.isfinite(sl) & (ew > 0) & (sl > 0)
        if ok.sum() < 3:
            continue
        score = np.median(np.abs(np.log10(ew[ok] / sl[ok])))
        if score < best_score:
            best_score, best = score, float(ff)
    return best
f_bound = _frequency_handoff()
print('frequency long->short handoff f_bound = %.3f Hz (trusted band <= %.1f Hz)' % (f_bound, f_high))
# Saturation form (2*pi*f)^5 g^-2 F(f) [rad]; both branches use elevation spectrum F(f).
B_f_ewdm_binned = (2*np.pi*np.reshape(f_ewdm,(1,len(f_ewdm))))**5/g**2*F_f_ewdm_binned
B_f_slope_binned = (2*np.pi*np.reshape(f_Hz_slope,(1,len(f_Hz_slope))))**5/g**2*F_f_slope_binned

# Wavenumber: EWDM is elevation; direct slope -> elevation via /k^2
F_k_slope_binned = np.reshape(k_rad_m_slope,(1,len(k_rad_m_slope)))**-2*S_k_slope_binned
B_k_ewdm_binned = np.reshape(k_ewdm,(1,len(k_ewdm)))**3*F_k_ewdm_binned
B_k_slope_binned = np.reshape(k_rad_m_slope,(1,len(k_rad_m_slope)))*S_k_slope_binned

cmap = plt.get_cmap('cividis')
colors = [cmap(j) for j in np.linspace(0,1, len(U_centers))]

# Axis limits
f_lims = [1e-2,2e1]
Ff_lims = [1e-10,1e1]
Bf_lims = [1e-4,2e-1]

k_lims = [1e-2,1e3]
Fk_lims = [1e-12,1e1]
Bk_lims = [1e-5,2e-2]

alpha_faded = 0.30   # opacity of each technique beyond its trusted side of the boundary

f_eq_lims = [2e-1,7e-1]
f_sat_lims = [4e-1,1e0]

k_eq_lims = [2e-1,1.5e0]
k_sat_lims = [5e-1,1.5e1]

lw_thick = 2.5
lw_thin = 1.5

fig, axs = plt.subplots(2, 2, figsize=(fullwidth, fullwidth))

# Panel spec: (axis, x-grid, Y(nbins,nx), boundary, is_ewdm, low_bound, high_bound)
panels = [
    (axs[0,0], f_ewdm, F_f_ewdm_binned, f_bound, True, f_low, f_high),
    (axs[0,0], f_Hz_slope, F_f_slope_binned, f_bound, False, None, f_high),
    (axs[1,0], f_ewdm, B_f_ewdm_binned, f_bound, True, f_low, f_high),
    (axs[1,0], f_Hz_slope, B_f_slope_binned, f_bound, False, None, f_high),
    (axs[0,1], k_ewdm, F_k_ewdm_binned, k_bound, True, k_low, None),
    (axs[0,1], k_rad_m_slope, F_k_slope_binned, k_bound, False, None, None),
    (axs[1,1], k_ewdm, B_k_ewdm_binned, k_bound, True, k_low, None),
    (axs[1,1], k_rad_m_slope, B_k_slope_binned, k_bound, False, None, None),
]


def trusted_segments(x, bound, is_ewdm, low_bound=None, high_bound=None):
    # Solid (trusted) and faded (beyond) masks, overlapping by one point.
    # EWDM: trusted in [low_bound, bound]; faded tail clipped at high_bound (frequency column).
    # Direct slope: trusted at/above bound, solid through full short-wave range.
    x = np.asarray(x)
    if is_ewdm:
        solid = (x >= (low_bound if low_bound is not None else -np.inf)) & (x <= bound)
    else:
        solid = x >= bound
    faded = []
    if solid.any():
        i0, i1 = np.where(solid)[0][[0, -1]]
        if i0 > 0:
            m = np.zeros(len(x), bool)
            m[:i0+1] = True
            faded.append(m)
        if i1 < len(x)-1:
            m = np.zeros(len(x), bool)
            m[i1:] = True
            if is_ewdm and high_bound is not None:
                m &= (x <= high_bound)        # clip EWDM fade at gravity-band ceiling
            if m.sum() > 1:
                faded.append(m)
    return solid, faded


# Pass 1: black underlay, trusted (solid) segments only
for ax, x, Y, bound, is_ewdm, low_bound, high_bound in panels:
    x = np.asarray(x)
    solid, _ = trusted_segments(x, bound, is_ewdm, low_bound, high_bound)
    for i in np.arange(len(U_centers)):
        ax.loglog(x[solid], Y[i,solid], color='black', linewidth=lw_thick)

# Pass 2: colored, full opacity where trusted and faded beyond each boundary
for ax, x, Y, bound, is_ewdm, low_bound, high_bound in panels:
    x = np.asarray(x)
    solid, faded = trusted_segments(x, bound, is_ewdm, low_bound, high_bound)
    for i in np.arange(len(U_centers)):
        ax.loglog(x[solid], Y[i,solid], color=colors[i], linewidth=lw_thin)
        for fm in faded:
            ax.loglog(x[fm], Y[i,fm], color=colors[i], linewidth=lw_thin, alpha=alpha_faded)

# Saturation reference lines (Bjorkqvist et al. 2019): f^-5 / k^-3 slopes,
# universal constants alpha_f=2.5e-2*2*pi [rad], alpha_k=5.8e-3 [rad].
alpha_f, alpha_k = 2.5e-2*2*np.pi, 5.8e-3
# axs[0,0].loglog(f_eq_lims,1e-2*np.power(f_eq_lims,-4),'--',color='red',linewidth=lw_thin)
axs[0,0].loglog(f_sat_lims,alpha_f*g**2/np.power(2*np.pi,5)*np.power(f_sat_lims,-5),'--',color='red',linewidth=lw_thick)
# axs[1,0].loglog(f_eq_lims,4e-2*np.power(f_eq_lims,1),'--',color='red',linewidth=lw_thin)
axs[1,0].loglog(f_sat_lims,alpha_f*np.power(f_sat_lims,0),'--',color='red',linewidth=lw_thick)
# axs[0,1].loglog(k_eq_lims,5e-2*np.power(k_eq_lims,-2.5),'--',color='red',linewidth=lw_thin)
axs[0,1].loglog(k_sat_lims,alpha_k*np.power(k_sat_lims,-3),'--',color='red',linewidth=lw_thick)
# axs[1,1].loglog(k_eq_lims,2e-2*np.power(k_eq_lims,0.5),'--',color='red',linewidth=lw_thin)
axs[1,1].loglog(k_sat_lims,alpha_k*np.power(k_sat_lims,0),'--',color='red',linewidth=lw_thick)

# Axis labels and limits
axs[0,0].set(xlim=f_lims, ylim=Ff_lims, ylabel=r'F(f) [m$^2$Hz$^{-1}$]')
axs[1,0].set(xlim=f_lims, ylim=Bf_lims, ylabel=r'(2$\pi$f)$^5$g$^{-2}$F(f) [rad]', xlabel='f [Hz]')
axs[0,1].set(xlim=k_lims, ylim=Fk_lims, ylabel=r'F(k) [m$^3$]')
axs[1,1].set(xlim=k_lims, ylim=Bk_lims, ylabel=r'k$^3$F(k) [rad]', xlabel=r'k [rad m$^{-1}$]')

axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])

counter = 0
bound_per_col = [f_bound, k_bound]
low_per_col = [f_low, k_low]
for j in np.arange(2):
    for i in np.arange(2):
        axs[j,i].grid(which='major', linestyle='-', linewidth=0.75)
        axs[j,i].grid(which='minor', linestyle=':', linewidth=0.75)
        axs[j,i].axvline(bound_per_col[i], color='dimgray', linestyle='--', linewidth=1, alpha=0.7)
        axs[j,i].axvline(low_per_col[i], color='dimgray', linestyle='--', linewidth=1, alpha=0.7)
        axs[j,i].text(0.05,0.95,panel_labels[counter],fontsize=fsize,ha='center',va='center',transform=axs[j,i].transAxes)
        counter = counter + 1

plt.tight_layout()

norm = BoundaryNorm(U_boundaries, cmap.N)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axs, location='top', orientation='horizontal', fraction=0.05, pad=0.03,aspect=50)
cbar.set_label(r'U$_{10}$ [m s$^{-1}$]')

_omni_fig = os.environ.get('EPSS_OMNI_FIGNAME', 'omnidirectional_spectra_binned_by_wind')
plt.savefig('../_figures/%s.pdf' % _omni_fig, bbox_inches='tight')