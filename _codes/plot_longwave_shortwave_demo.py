"""
E-PSS elevation-reconstruction demo: long-wave path (Fourier slope projection on
disc-averaged slope timeseries) and short-wave path (per-frame g2s surface integration).
Stacked layout: slope timeseries -> long-wave eta(t) -> g2s short-wave eta(x,y) snapshot.
"""

import numpy as np
import netCDF4 as nc
from scipy import signal
from matplotlib import pyplot as plt

from subroutines.utils import figure_style, WATER_DEPTH_M, FS_HZ
from multiaperture import build_eta_field, fourier_slope_projection
color_list, fullwidth, fullheight, fsize = figure_style()

import warnings
warnings.filterwarnings("ignore")

path = '../_data/'
fs = FS_HZ
depth = WATER_DEPTH_M
run_ind = 180         # ~1 m Hs run; inferred long wave is in phase with the lidar here
slope_aperture = 32    # full-frame disc

# Slope-field stack for one run (earth-referenced; ny, nx, T)
fld = nc.Dataset(path+'ASIT2019_slope_fields_reduced.nc')
SxF = np.nan_to_num(np.ma.filled(fld['slope_east'][run_ind], np.nan)).astype(float)
SyF = np.nan_to_num(np.ma.filled(fld['slope_north'][run_ind], np.nan)).astype(float)
ny, nx, T = SxF.shape

# Split elevation into long-wave (Fourier slope projection) and short-wave (g2s) components
eta, dx = build_eta_field(SxF, SyF, depth, fs, slope_aperture=slope_aperture)
yy, xx = np.ogrid[:ny, :nx]
disc = (yy-(ny-1)/2.0)**2 + (xx-(nx-1)/2.0)**2 <= (slope_aperture/2.0)**2
sE_disc = SxF[disc].mean(0)            # disc-averaged slope timeseries (long-wave input)
sN_disc = SyF[disc].mean(0)
# subtract the long wave; Z is the pure g2s short-wave field
eta_long = fourier_slope_projection(SxF, SyF, depth, fs)
Z = eta - eta_long[None, None, :]      # g2s short-wave elevation field

# Riegl lidar: low-passed and phase-aligned to the inferred long-wave elevation
ds_o = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
lidar = np.ma.filled(ds_o['wse_m_Riegl'][0, :, run_ind], np.nan)
_b, _a = signal.butter(4, 0.4/(fs/2), 'low')
eta_lp = signal.filtfilt(_b, _a, np.nan_to_num(eta_long - np.nanmean(eta_long)))
lidar_lp = signal.filtfilt(_b, _a, np.nan_to_num(lidar - np.nanmean(lidar)))
_xc = signal.correlate(eta_lp, lidar_lp, mode='full')
_lags = signal.correlation_lags(len(eta_lp), len(lidar_lp), mode='full')
_sel = np.abs(_lags) <= int(5*fs)
lag = int(_lags[_sel][np.argmax(_xc[_sel])])
lidar_shift = np.roll(lidar_lp, lag)   # lidar aligned to the inferred long wave

# 10-s display window; snapshot at midpoint (t = 5 s)
t = np.arange(T)/fs
t_start = 310.0       # window where the inferred long wave best matches the lidar
i0 = int(t_start*fs)
i1 = i0 + int(10*fs)
i_snap = i0 + int(5*fs)
tw = t[i0:i1] - t_start
t_snap = tw[i_snap-i0]

sE_d = sE_disc - sE_disc.mean()
sN_d = sN_disc - sN_disc.mean()
xg = (np.arange(nx)-(nx-1)/2.0)*dx     # centered spatial axes [m]
yg = (np.arange(ny)-(ny-1)/2.0)*dx

slope_lim = 0.4
Zsnap = Z[:, :, i_snap]
vZ = np.percentile(np.abs(Zsnap), 99)          # symmetric coolwarm limit about eta = 0

fig = plt.figure(figsize=(fullwidth/2, fullwidth*1.05), constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 0.5, 1.05])
ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

def panel_tag(ax, tag):
    ax.text(0.07, 0.90, tag, transform=ax.transAxes, fontsize=fsize, va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=1.0))

# slope timeseries (long-wave input)
ax0.plot(tw, sE_d[i0:i1], '--', color='black', lw=2, label=r'$s_E$')
ax0.plot(tw, sN_d[i0:i1], ':', color='black', lw=2, label=r'$s_N$')
ax0.set_ylabel('slope [rad]')
ax0.set_ylim(-slope_lim, slope_lim)
ax0.set_xlim(0, 10)
ax0.legend(loc='upper right', fontsize=fsize)
ax0.axvline(t_snap, color=color_list[3], lw=1.5, alpha=0.6)
ax0.set_xticklabels([])
panel_tag(ax0, '(a)')

# long-wave elevation (Fourier slope projection) with the phase-aligned lidar overlay
ax1.plot(tw, eta_long[i0:i1], color=color_list[2], lw=2, label='E-PSS')
ax1.plot(tw, lidar_shift[i0:i1], 'k--', lw=2, label=r'lidar') #label=r'lidar (%+.1f s)' % (lag/fs))
ax1.set_ylabel(r'long wave $\eta$ [m]')
ax1.set_xlim(0, 10)
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel('t [s]')
ax1.axvline(t_snap, color=color_list[3], lw=1.5, alpha=0.6)
ax1.legend(loc='lower right', fontsize=fsize)
panel_tag(ax1, '(b)')

# g2s short-wave elevation field at t = 5 s (full width, colorbar atop)
im = ax2.pcolormesh(xg, yg, Zsnap, cmap='coolwarm', vmin=-vZ, vmax=vZ, shading='auto', rasterized=True)
ax2.set_aspect('equal')
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
cb = fig.colorbar(im, ax=ax2, orientation='horizontal', location='top', fraction=0.047, pad=0.02)
cb.set_label(r'short wave $\eta$ [m], t = %.0f s' % t_snap)
panel_tag(ax2, '(c)')

plt.savefig('../_figures/longwave_shortwave_demo_stacked.pdf', bbox_inches='tight', dpi=300)
