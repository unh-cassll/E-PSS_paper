"""
Resolution of the long-wave phase ambiguity from surface curvature, 2 x 2.

  top row     instantaneous phase of the reconstruction relative to the lidar
  bottom row  long-wave elevation: lidar (black dashed), E-PSS before the
              curvature fix (gray), after it (dark crimson)
  columns     two long-wave directions. The projection phase takes
              cos(theta_f) = |S_E|/M, which is non-negative by construction, so
              it can only point east: it is already correct for an east-going
              sea (left) and inverted for a west-going one (right), where the
              curvature vote repairs it.

The lidar is aligned to the POST-fix series only and the same lag is applied to
the pre-fix trace, so the alignment cannot absorb the 180-degree flip.

@author: nathanlaxague
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from subroutines.utils import figure_style, WATER_DEPTH_M, FS_HZ
from multiaperture.pss import fourier_slope_projection
color_list, fullwidth, fullheight, fsize = figure_style()

import warnings
warnings.filterwarnings("ignore")

path = '../_data/'
figpath = '../_figures/'
fs, depth = FS_HZ, WATER_DEPTH_M
BAND = (0.08, 0.30)          # Hz, long-wave band the phase is read over
# (run, display window [s]) per column; each window is shifted to start at zero
# so both columns read 0-30 s
CASES = [(159, (20.0, 50.0)),
         (111, (22.0, 52.0))]
WIN = 60.0                   # s of record the display window is drawn from

C_LIDAR, C_PRE, C_POST = 'k', '0.62', color_list[2]
C_PRE_TXT = '0.38'           # darker than the trace so the annotation reads clearly

fld = nc.Dataset(path+'ASIT2019_slope_fields_reduced.nc')
env = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
ds = xr.open_dataset(path+'ASIT2019_EPSS_directional_spectra.nc')
om = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')
sos = signal.butter(4, [BAND[0]/(fs/2), BAND[1]/(fs/2)], 'band', output='sos')

U10 = np.ma.filled(env['U10_best'][:], np.nan)
WDIR = np.ma.filled(env['COARE_Wdir'][:], np.nan)      # coming-from, deg CW from N


def mean_dir(run):
    """Energy-weighted long-wave mean direction [deg CW from N, going-to]."""
    f = ds['frequency'].values
    b = (f >= BAND[0]) & (f <= BAND[1])
    th = ds['mean_direction'][:, run].values[b]
    w = ds['S_f'][:, run].values[b]
    g = np.isfinite(th) & np.isfinite(w)
    if not g.any():
        return np.nan
    a = th[g]
    return float(np.degrees(np.arctan2(np.nansum(w[g]*np.sin(a)),
                                       np.nansum(w[g]*np.cos(a)))) % 360.0)


def conditions(run):
    """(U10 [m/s], lidar Hm0 [m], wind going-to [deg], wave going-to [deg])."""
    f = om['frequency'].values
    b = (f > 0.08) & (f < 0.50)
    hm0 = 4*np.sqrt(np.nansum(om['F_f_m2_Hz_lidar'].values[b, run])
                    * np.median(np.diff(f)))
    return U10[run], hm0, (WDIR[run] + 180.0) % 360.0, mean_dir(run)


def prep(run):
    """Band-passed lidar and the two reconstructions, plus their phase relative
    to the lidar and correlation against it."""
    se = np.nan_to_num(np.ma.filled(fld['slope_east'][run], np.nan)).astype(float)
    sn = np.nan_to_num(np.ma.filled(fld['slope_north'][run], np.nan)).astype(float)
    pre = fourier_slope_projection(se, sn, depth, fs, phase_source='projection')
    post = fourier_slope_projection(se, sn, depth, fs, phase_source='curvature')
    lid = np.ma.filled(env['wse_m_Riegl'][0, :, run], np.nan)
    n = min(len(lid), len(pre))
    Lb = signal.sosfiltfilt(sos, np.nan_to_num(lid[:n] - np.nanmean(lid[:n])))
    Pb = signal.sosfiltfilt(sos, pre[:n])
    Cb = signal.sosfiltfilt(sos, post[:n])
    xc = signal.correlate(Cb, Lb, mode='full')
    lg = signal.correlation_lags(len(Cb), len(Lb), mode='full')
    s = np.abs(lg) <= int(6*fs)
    La = np.roll(Lb, int(lg[s][np.argmax(xc[s])]))
    aL = signal.hilbert(La)
    dphi = lambda x: np.degrees(np.angle(signal.hilbert(x)*np.conj(aL)))
    r = lambda x: float(np.corrcoef(x, La)[0, 1])
    return dict(t=np.arange(n)/fs, lid=La, pre=Pb, post=Cb,
                dpre=dphi(Pb), dpost=dphi(Cb), rpre=r(Pb), rpost=r(Cb))


fig, axs = plt.subplots(2, 2, figsize=(fullwidth, fullwidth*0.50),
                        sharex='col', sharey='row', constrained_layout=True)
tags = [['(a)', '(b)'], ['(c)', '(d)']]
ymax = 0.0

for col, (run, (ta, tb)) in enumerate(CASES):
    d = prep(run)
    # window centered where the post-fix trace is most energetic
    env_p = np.abs(signal.hilbert(d['post']))
    k = np.argmax(np.convolve(env_p, np.ones(int(WIN*fs))/(WIN*fs), 'same'))
    i0 = max(0, k - int(WIN*fs/2))
    i0, i1 = i0 + int(ta*fs), i0 + int(tb*fs)
    tw = d['t'][i0:i1] - d['t'][i0]

    ax = axs[0, col]
    ax.plot(tw, d['dpre'][i0:i1], '-', color=C_PRE, lw=1.6)
    ax.plot(tw, d['dpost'][i0:i1], '-', color=C_POST, lw=1.8)
    for y in (-180, 0, 180):
        ax.axhline(y, color='k', ls=':', lw=0.9, alpha=0.5)
    ax.set_ylim(-200, 200); ax.set_yticks([-180, -90, 0, 90, 180])
    u, h, wd, sd = conditions(run)
    ax.set_title(f'$U_{{10}}$={u:.1f} m s$^{{-1}}$, $H_{{m0}}$={h:.2f} m, '
                 f'wind {wd:.0f}$^\\circ$, waves {sd:.0f}$^\\circ$',
                 fontsize=fsize-2)
    if col == 0:
        ax.set_ylabel(r'phase rel. lidar [$^\circ$]')
    ax.text(0.015, 0.90, tags[0][col], transform=ax.transAxes, fontsize=fsize,
            va='center', bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.9))

    ax = axs[1, col]
    ax.plot(tw, d['lid'][i0:i1], '--', color=C_LIDAR, lw=2.0)
    ax.plot(tw, d['pre'][i0:i1], '-', color=C_PRE, lw=1.6)
    ax.plot(tw, d['post'][i0:i1], '-', color=C_POST, lw=1.8)
    ax.set_xlabel('t [s]'); ax.set_xlim(0, tb - ta)
    if col == 0:
        ax.set_ylabel(r'$\eta_{\rm long}$ [m]')
    ymax = max(ymax, np.nanmax(np.abs(np.concatenate(
        [d['lid'][i0:i1], d['pre'][i0:i1], d['post'][i0:i1]]))))
    # correlation against the lidar, color-coded to its trace
    for yy, rv, cc in ((0.030, d['rpost'], C_POST), (0.170, d['rpre'], C_PRE_TXT)):
        ax.text(0.985, yy, r'$r=%+.2f$' % rv, transform=ax.transAxes,
                color=cc, fontsize=fsize-2, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none',
                          alpha=0.75))
    ax.text(0.015, 0.90, tags[1][col], transform=ax.transAxes, fontsize=fsize,
            va='center', bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.9))

axs[1, 0].set_ylim(-1.35*ymax, 1.35*ymax)          # sharey='row' carries this to (d)

axs[0, 0].legend(handles=[Line2D([], [], color=C_LIDAR, ls='--', lw=2.0, label='lidar'),
                          Line2D([], [], color=C_PRE, lw=1.6, label='E-PSS, before fix'),
                          Line2D([], [], color=C_POST, lw=1.8, label='E-PSS, after fix')],
                 loc='upper right', fontsize=fsize-3, framealpha=0.92)

fig.savefig(figpath+'phase_elevation_pair.pdf', bbox_inches='tight', dpi=300)
